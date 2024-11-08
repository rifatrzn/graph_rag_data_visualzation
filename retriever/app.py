from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from nomic import embed  # Nomic Python API for embeddings
from neo4j import GraphDatabase
import getpass
from langchain_nomic import NomicEmbeddings

# Load environment variables from .env file
load_dotenv()

# Prompt for Nomic API key if it's not set in environment
if not os.getenv("NOMIC_API_KEY"):
    os.environ["NOMIC_API_KEY"] = getpass.getpass("Enter your Nomic API key: ")

# Ensure Nomic API key is set
nomic_api_key = os.getenv("NOMIC_API_KEY")
if not nomic_api_key:
    raise ValueError("NOMIC_API_KEY is not set. Please provide a valid Nomic API key.")

# Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://d1cf7d0d.databases.neo4j.io")  # Default URI for the database
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "<Username>")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "<Password>")

if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    raise ValueError("Neo4j credentials are not properly set in the environment.")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Verify Neo4j connectivity
try:
    with driver.session() as session:
        driver.verify_connectivity()
        print("Connected to Neo4j successfully!")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    driver.close()
    exit()
    

# Function to generate Nomic embeddings using Nomic API
def get_nomic_embedding(text):
    try:
        # Generate the embeddings using Nomic API
        response = embed.text(
            texts=[text],  # Embed the text as a list
            model="nomic-embed-text-v1.5",  # Specify the model name
            task_type="search_query"  # Use appropriate task type
        )
        return response["embeddings"][0]  # Return the first embedding
    except Exception as e:
        raise ValueError(f"Error in generating embeddings: {str(e)}")

# Example query
question = "give me a list of healthcare providers in the area of dermatology"


# Generate Nomic embeddings for the query
try:
    question_embedding = get_nomic_embedding(question)
except ValueError as e:
    print(e)
    driver.close()
    exit()


# Query Neo4j database using Nomic embeddings
def query_neo4j(embedding):
    try:
        with driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes(
                    'health_providers_embeddings',
                    $top_k,
                    $question_embedding
                ) YIELD node AS healthcare_provider, score
                RETURN healthcare_provider.name, healthcare_provider.bio, score
                """,
                question_embedding=embedding,
                top_k=3,  # Number of top results to retrieve
            )
            return result
    except Exception as e:
        print(f"Error querying Neo4j: {str(e)}")
        driver.close()
        exit()


# Run Neo4j query and print results
try:
    result = query_neo4j(question_embedding)
    for record in result:
        print(f"Name: {record['healthcare_provider.name']}")
        print(f"Bio: {record['healthcare_provider.bio']}")
        print(f"Score: {record['score']}")
        print("---")
finally:
    driver.close()  # Ensure the driver is closed