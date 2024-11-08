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
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_community.document_loaders import UnstructuredFileLoader
from pydantic import BaseModel, Field




from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_nvidia_ai_endpoints import ChatNVIDIA



load_dotenv()


AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")





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

def create_enhanced_visualization_data(tx):
    # Create category and data type nodes
    setup_query = """
    // Create visualization categories
    MERGE (time:VisualizationCategory {name: 'Time Series'})
    MERGE (distribution:VisualizationCategory {name: 'Distribution'})
    MERGE (comparison:VisualizationCategory {name: 'Comparison'})
    MERGE (geographical:VisualizationCategory {name: 'Geographical'})
    MERGE (relationship:VisualizationCategory {name: 'Relationship'})
    
    // Create data type nodes
    MERGE (categorical:DataType {name: 'Categorical'})
    MERGE (numerical:DataType {name: 'Numerical'})
    MERGE (temporal:DataType {name: 'Temporal'})
    MERGE (spatial:DataType {name: 'Spatial'})
    """
    tx.run(setup_query)

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
    # Create relationships between similar visualization types and queries
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

def load_enhanced_graph():
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # Create the basic structure
            session.execute_write(create_enhanced_visualization_data)
            # Create additional relationships
            session.execute_write(create_visualization_relationships)
            print("Enhanced graph structure successfully created")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    load_enhanced_graph()
    