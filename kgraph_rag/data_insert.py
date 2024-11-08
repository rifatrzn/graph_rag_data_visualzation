# Import necessary modules
import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Function to read CSV data
def read_csv_data(file_path):
    return pd.read_csv(file_path)

# Function to create nodes and relationships in Neo4j
def create_graph_from_csv(tx, data):
    for _, row in data.iterrows():
        tx.run("""
        MERGE (ct:ChartType {name: $chart_type})
        MERGE (dt:DataType {name: $data_type})
        MERGE (ct)-[:SUITABLE_FOR]->(dt)
        
        CREATE (usage:UseCase {description: $best_use_case})
        CREATE (practice:BestPractice {description: $best_practice})
        CREATE (mistake:CommonMistake {description: $common_mistake})
        CREATE (palette:ColorPalette {description: $color_palette})
        CREATE (example:ExampleUsage {description: $example_usage})
        
        MERGE (ct)-[:USED_IN]->(usage)
        MERGE (ct)-[:BEST_PRACTICE]->(practice)
        MERGE (ct)-[:AVOID]->(mistake)
        MERGE (ct)-[:COLOR_GUIDE]->(palette)
        MERGE (ct)-[:EXAMPLE]->(example)
        """, 
        {
            "chart_type": row['Chart Type'],
            "data_type": row['Data Type'],
            "best_use_case": row['Best Use Case'],
            "best_practice": row['Best Practice'],
            "common_mistake": row['Common Mistakes'],
            "color_palette": row['Color Palette'],
            "example_usage": row['Example Usage']
        })

# Main function to read CSV and create graph data
def load_csv_to_graph(file_path):
    data = read_csv_data(file_path)
    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(lambda tx: create_graph_from_csv(tx, data))
    print("CSV data has been imported and structured in Neo4j.")

if __name__ == "__main__":
    csv_file_path = "./BestTypeData.csv"  # Replace with your CSV file path
    load_csv_to_graph(csv_file_path)
    driver.close()
