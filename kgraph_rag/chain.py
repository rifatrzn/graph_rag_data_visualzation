import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Tuple
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
import requests
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph

import pandas as pd
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import pydantic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.text_splitter import TokenTextSplitter

from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional

from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Set up credentials and connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
OLLAMA_BASE_URL = "http://ollama:11434/"
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



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")




# Ensure NVIDIA_API_KEY is loaded
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in environment variables")
print(os.getenv("NVIDIA_API_KEY")) 
# Initialize NVIDIA Chat Model
chat = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    api_key=NVIDIA_API_KEY,
    temperature=0.1,
    top_p=0.7,
    max_tokens=1024
)




# JSON data we will use for this example
data = [
    {
        "query": "Distribution of Plans by Metal Level",
        "best_type_of_graph": "Bar Chart",
        "reason": "A bar chart effectively compares the frequency of different categories (metal levels) in a clear, straightforward manner."
    },
    {
        "query": "Average Premiums by State",
        "best_type_of_graph": "Choropleth Map or Bar Chart",
        "reason": "A choropleth map provides a geographical visualization of data, highlighting variations across regions. A bar chart is also suitable for comparing values across different states."
    },
    {
        "query": "Plan Availability by County",
        "best_type_of_graph": "Heat Map",
        "reason": "A heat map can represent the density of available plans across counties, making it easier to identify areas with higher or lower availability."
    },
    {
        "query": "Out-of-Pocket Costs by Plan Type",
        "best_type_of_graph": "Box Plot",
        "reason": "Box plots are ideal for showing the distribution of costs, highlighting the median, quartiles, and outliers within each plan type."
    },
    {
        "query": "Issuer Market Share",
        "best_type_of_graph": "Pie Chart",
        "reason": "A pie chart visually represents parts of a whole, making it suitable for showing the proportion of the market each issuer holds."
    },
    {
        "query": "Rate of Increase in Premiums Over Years",
        "best_type_of_graph": "Line Graph",
        "reason": "A line graph is perfect for showing trends over time, such as how premiums have increased or decreased across different years."
    },
    {
        "query": "Distribution of Specific Benefits (e.g., dental coverage) by Plan",
        "best_type_of_graph": "Stacked Bar Chart",
        "reason": "Stacked bar charts can show the proportion of plans with specific benefits, allowing for comparison across different categories simultaneously."
    },
    {
        "query": "Comparison of Deductibles Across States",
        "best_type_of_graph": "Box Plot or Violin Plot",
        "reason": "These plots can display the distribution and range of deductibles across states, highlighting variations and trends."
    },
    {
        "query": "Impact of Metal Level on Deductibles and Out-of-Pocket Maximums",
        "best_type_of_graph": "Grouped Bar Chart",
        "reason": "Grouped bar chart or side-by-side bar chart would allow us to compare two metrics (average deductible and average out-of-pocket maximum) across the different metal levels in a single visualization."
    },
    {
        "query": "Relationship Between Premiums and Out-of-Pocket Maximums",
        "best_type_of_graph": "Scatter Plot",
        "reason": "A scatter plot is useful for examining the relationship between two quantitative variables, revealing patterns or correlations."
    }
]


# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def create_indexes(tx):
    create_index_query = """
    CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (n:Query|VisualizationType|VisualizationCategory|DataType)
    ON EACH [n.name, n.reason];
    """
    tx.run(create_index_query)
    print("Index 'entity' created or verified.")

def setup_categories(tx):
    setup_query = """
    // Create visualization categories and data type nodes
    MERGE (time:VisualizationCategory {name: 'Time Series'})
    MERGE (distribution:VisualizationCategory {name: 'Distribution'})
    MERGE (comparison:VisualizationCategory {name: 'Comparison'})
    MERGE (geographical:VisualizationCategory {name: 'Geographical'})
    MERGE (relationship:VisualizationCategory {name: 'Relationship'})
    MERGE (categorical:DataType {name: 'Categorical'})
    MERGE (numerical:DataType {name: 'Numerical'})
    MERGE (temporal:DataType {name: 'Temporal'})
    MERGE (spatial:DataType {name: 'Spatial'})
    """
    tx.run(setup_query)
    
def create_visualization_data(tx, item):
    cypher_query = """
    // Create the main query node
    MERGE (q:Query {name: $query_name})
    
    // Create or match the visualization type
    MERGE (v:VisualizationType {name: $viz_type})
    
    // Create the main relationship with reason
    MERGE (q)-[r:BEST_VISUALIZED_AS {reason: $viz_reason}]->(v)
    
    // Connect to appropriate categories
    WITH q, v
    MATCH (cat:VisualizationCategory)
    WHERE 
        (cat.name = 'Time Series' AND $query_name CONTAINS 'Over Years') OR
        (cat.name = 'Distribution' AND ($query_name CONTAINS 'Distribution' OR $viz_type CONTAINS 'Box Plot')) OR
        (cat.name = 'Geographical' AND ($query_name CONTAINS 'State' OR $query_name CONTAINS 'County')) OR
        (cat.name = 'Comparison' AND $viz_type CONTAINS 'Bar Chart') OR
        (cat.name = 'Relationship' AND $viz_type CONTAINS 'Scatter Plot')
    
    MERGE (v)-[:BELONGS_TO]->(cat)
    
    // Connect to data types
    WITH q, v
    MATCH (dt:DataType)
    WHERE 
        (dt.name = 'Categorical' AND ($query_name CONTAINS 'Type' OR $query_name CONTAINS 'Level')) OR
        (dt.name = 'Numerical' AND ($query_name CONTAINS 'Costs' OR $query_name CONTAINS 'Premiums')) OR
        (dt.name = 'Temporal' AND $query_name CONTAINS 'Years') OR
        (dt.name = 'Spatial' AND ($query_name CONTAINS 'State' OR $query_name CONTAINS 'County'))
    
    MERGE (q)-[:USES_DATA_TYPE]->(dt)
    """
    
    tx.run(cypher_query, 
           query_name=item["query"], 
           viz_type=item["best_type_of_graph"], 
           viz_reason=item["reason"])

    # Process each item in data
    for item in data:
        cypher_query = """
        // Create the main query node
        MERGE (q:Query {name: $query_name})
        
        // Create or match the visualization type
        MERGE (v:VisualizationType {name: $viz_type})
        
        // Create the main relationship with reason
        MERGE (q)-[r:BEST_VISUALIZED_AS {reason: $viz_reason}]->(v)
        
        // Connect to appropriate categories
        WITH q, v
        MATCH (cat:VisualizationCategory)
        WHERE 
            (cat.name = 'Time Series' AND $query_name CONTAINS 'Over Years') OR
            (cat.name = 'Distribution' AND ($query_name CONTAINS 'Distribution' OR $viz_type CONTAINS 'Box Plot')) OR
            (cat.name = 'Geographical' AND ($query_name CONTAINS 'State' OR $query_name CONTAINS 'County')) OR
            (cat.name = 'Comparison' AND $viz_type CONTAINS 'Bar Chart') OR
            (cat.name = 'Relationship' AND $viz_type CONTAINS 'Scatter Plot')
        
        MERGE (v)-[:BELONGS_TO]->(cat)
        
        // Connect to data types
        WITH q, v
        MATCH (dt:DataType)
        WHERE 
            (dt.name = 'Categorical' AND ($query_name CONTAINS 'Type' OR $query_name CONTAINS 'Level')) OR
            (dt.name = 'Numerical' AND ($query_name CONTAINS 'Costs' OR $query_name CONTAINS 'Premiums')) OR
            (dt.name = 'Temporal' AND $query_name CONTAINS 'Years') OR
            (dt.name = 'Spatial' AND ($query_name CONTAINS 'State' OR $query_name CONTAINS 'County'))
        
        MERGE (q)-[:USES_DATA_TYPE]->(dt)
        """
        
        tx.run(cypher_query, 
               query_name=item["query"], 
               viz_type=item["best_type_of_graph"], 
               viz_reason=item["reason"])        
        
        

def create_visualization_relationships(tx):
    relationship_query = """
    // Connect similar visualization types
    MATCH (v1:VisualizationType)
    WITH v1
    MATCH (v2:VisualizationType)
    WHERE v1 <> v2 AND (
        (v1.name CONTAINS 'Bar' AND v2.name CONTAINS 'Bar') OR
        (v1.name CONTAINS 'Plot' AND v2.name CONTAINS 'Plot')
    )
    MERGE (v1)-[:SIMILAR_TO]->(v2)
    WITH v1, v2  // Add WITH to ensure continuation

    // Connect queries with similar topics
    MATCH (q1:Query)
    WITH q1
    MATCH (q2:Query)
    WHERE q1 <> q2 AND (
        (q1.name CONTAINS 'Premiums' AND q2.name CONTAINS 'Premiums') OR
        (q1.name CONTAINS 'State' AND q2.name CONTAINS 'State') OR
        (q1.name CONTAINS 'Plan' AND q2.name CONTAINS 'Plan')
    )
    WITH q1, q2  // Another WITH before final MERGE
    MERGE (q1)-[:RELATED_TO]->(q2)
    """
    tx.run(relationship_query)

def load_enhanced_graph(uri, username, password, database):
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session(database=database) as session:
            # Create indexes
            session.execute_write(create_indexes)
            
            # Set up categories
            session.execute_write(setup_categories)
            
            # Create visualization data
            for item in data:
                session.execute_write(lambda tx: create_visualization_data(tx, item))
            
            # Create relationships
            session.execute_write(create_visualization_relationships)
    
    print("Enhanced graph structure successfully created")

# Usage
load_enhanced_graph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

    


# Initialize embeddings properly
class OllamaEmbeddings:
    def __init__(self, endpoint, model="nomic-embed-text"):
        self.endpoint = endpoint
        self.model = model

    def embed_query(self, text):
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json().get("embedding")

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

ollama_embeddings = OllamaEmbeddings(EMBEDDINGS_ENDPOINT, model="nomic-embed-text")

try:
    # Initialize vector index using a correct embedding instance
    vector_index = Neo4jVector.from_existing_graph(
        embedding=ollama_embeddings,
        graph=kg,
        node_label="Query",
        text_node_properties=["name", "reason"],
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
        ..., description="All the person, organization, or business entities that appear in the text"
    )

# Prompt for entity extraction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
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
                CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
                YIELD node, score
                CALL {
                  WITH node
                  MATCH (node)-[r:MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50""",
                {"query": generate_full_text_query(entity)}
            )
            result += "\n".join([el["output"] for el in response])
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
