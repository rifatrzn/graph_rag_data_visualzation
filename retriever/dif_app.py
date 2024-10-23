from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.utilities import Neo4j
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

# Add Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize Nomic embeddings with Ollama
nomic_embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v1",
    base_url=OLLAMA_BASE_URL
)

# Example usage of the embeddings
def get_embeddings(text):
    return nomic_embeddings.embed_query(text)


# Create a simple vector database
texts = ["This is a sample text", "Another example", "Yet another sample"]
embeddings = [get_embeddings(text) for text in texts]
vector_db = FAISS.from_embeddings(embeddings, texts, nomic_embeddings)

# Perform a similarity search
query = "Find me a sample"
results = vector_db.similarity_search(query)
print(results)


def add_embeddings_to_neo4j(text, node_label):
    embedding = get_embeddings(text)
    
    with GraphDatabase.driver(NEO4J_URI, auth=AUTH) as driver:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                MERGE (n:%s {text: $text})
                SET n.embedding = $embedding
            """ % node_label, text=text, embedding=embedding)

# Example usage
add_embeddings_to_neo4j("This is a sample text", "TextNode")


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ... existing code ...

# Assuming you've created a vector_db as in the first example
retriever = vector_db.as_retriever()

llm = ChatOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Example usage
query = "What is an example of a sample text?"
response = qa_chain.run(query)
print(response)

from sklearn.cluster import KMeans
import numpy as np

# ... existing code ...

def cluster_texts(texts, n_clusters=3):
    embeddings = [get_embeddings(text) for text in texts]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_

# Example usage
texts = ["This is a sample", "Another example", "Yet another sample", "Something different", "Totally unrelated"]
clusters = cluster_texts(texts)
for text, cluster in zip(texts, clusters):
    print(f"Text: {text} | Cluster: {cluster}")
    
    
import numpy as np

# ... existing code ...

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend_similar_items(item, item_list, top_n=3):
    item_embedding = get_embeddings(item)
    similarities = []
    for other_item in item_list:
        other_embedding = get_embeddings(other_item)
        similarity = cosine_similarity(item_embedding, other_embedding)
        similarities.append((other_item, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Example usage
items = ["Red shirt", "Blue jeans", "Green sweater", "Black shoes", "White socks"]
target_item = "Blue t-shirt"
recommendations = recommend_similar_items(target_item, items)
print(f"Recommendations for '{target_item}':")
for item, similarity in recommendations:
    print(f"- {item} (similarity: {similarity:.2f})")