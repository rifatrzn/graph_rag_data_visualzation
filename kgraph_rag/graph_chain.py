import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Load environment variables
load_dotenv()

# Set up credentials and connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
EMBEDDINGS_ENDPOINT = "http://localhost:11434/api/embeddings"

# Initialize Neo4j connection
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

print(f"Attempting to connect to Neo4j at: {NEO4J_URI}")

try:
    with driver.session() as session:
        result = session.run("RETURN 1 AS num")
        print(f"Successfully connected to Neo4j. Test query result: {result.single()['num']}")
except Exception as e:
    print(f"Failed to connect to Neo4j: {str(e)}")

# Function to create full-text index
def create_fulltext_index(driver):
    with driver.session() as session:
        session.run("""
        CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (n:ChartType|DataType|UseCase|BestPractice)
        ON EACH [n.name, n.description];
        """)
        print("Fulltext index 'entity' created or verified.")

# Create the index
create_fulltext_index(driver)

# Initialize vector index using embedding model
try:
    vector_index = Neo4jVector.from_existing_graph(
        embedding=EMBEDDINGS_ENDPOINT,
        graph=kg,
        node_label="ChartType",
        text_node_properties=["name", "description"],
        embedding_node_property="embedding",
        index_name="visualization_index"
    )
    print("Successfully created Neo4jVector")
except Exception as e:
    print(f"Error creating vector index: {e}")

# Define entity extraction class
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ..., description="All the chart, data type, or use case entities that appear in the text"
    )

# Prompt for entity extraction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting relevant visualization and data type entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}"),
])
entity_chain = prompt | ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=None).with_structured_output(Entities)

# Generate full-text query
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Structured retriever function
def structured_retriever(question: str) -> str:
    try:
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            print(f"Getting Entity: {entity}")
            response = kg.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit: 5})
                YIELD node, score
                RETURN node.name AS name, node.description AS description, score
                """,
                {"query": generate_full_text_query(entity)}
            )
            result += "\n".join([f"{el['name']} ({el['description']}) - Score: {el['score']}" for el in response])
        return result if result else "No structured data found."
    except Exception as e:
        print(f"Error in structured retriever: {e}")
        return f"Error in structured retrieval: {str(e)}"

# Final retriever function
def retriever(question: str) -> str:
    try:
        print(f"Search query: {question}")
        structured_data = structured_retriever(question)
        print(f"Structured data retrieved: {structured_data}")
        
        unstructured_data = []
        try:
            unstructured_data = [
                el.page_content for el in vector_index.similarity_search(question)
            ]
        except Exception as e:
            print(f"Error during vector search: {e}")
            unstructured_data.append(f"Error during vector search: {str(e)}")
        
        final_data = f"Structured data:\n{structured_data}\nUnstructured data:\n{'#Document '.join(unstructured_data)}"
        print(f"\nFinal Data::: ==> {final_data}")
        return final_data
    except Exception as e:
        print(f"Retriever error: {e}")
        return f"Error retrieving data: {str(e)}"

# Define the RAG chain
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=None)
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=None)
    | StrOutputParser()
)

# Test the RAG chain
res_simple = chain.invoke({"question": "What is the best type of graph for comparing premiums by state?"})
print(f"\nResults === {res_simple}\n\n")
