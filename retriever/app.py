from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from nomic import atlas  # Nomic Python API for embeddings
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]

# Initialize Neo4j graph connection
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# Initialize Nomic client (ensure API key is set)
nomic_api_key = os.getenv("NOMIC_API_KEY")
atlas.login(nomic_api_key)  # Make sure you have your Nomic API key set in .env

# Function to generate Nomic embeddings using Nomic API
def get_nomic_embedding(text):
    # Generate the embeddings using Nomic Atlas
    embedding = atlas.embed_text(text)  # Embed the text
    return embedding[0]  # Return the first embedding

# Example question to encode
question = "give me a list of healthcare providers in the area of dermatology"

# Generate Nomic embeddings for the question
question_embedding = get_nomic_embedding(question)

# Query the Neo4j database using Nomic embeddings
result = kg.query(
    """
    CALL db.index.vector.queryNodes(
        'health_providers_embeddings',
        $top_k,
        $question_embedding
    ) YIELD node AS healthcare_provider, score
    RETURN healthcare_provider.name, healthcare_provider.bio, score
    """,
    params={
        "question_embedding": question_embedding,  # Use the generated Nomic embedding
        "top_k": 3,
    },
)

# Print the results
for record in result:
    print(f"Name: {record['healthcare_provider.name']}")
    print(f"Bio: {record['healthcare_provider.bio']}")
    print(f"Score: {record['score']}")
    print("---")