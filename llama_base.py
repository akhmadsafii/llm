import os
import json
import requests
import logging
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

# Function to retrieve data from API endpoint
def retrieve_from_endpoint(url: str) -> dict:
    """Retrieve data from the API endpoint."""
    headers = {"Authorization": SECTORS_API_KEY}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the JSON object directly
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving data from {url}: {e}")
        return {"error": str(e)}

def format_date(date: str) -> str:
    """Format the date to YYYY-MM-DD. If invalid, return today's date."""
    try:
        # Try parsing the date in various formats
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        try:
            # Try parsing other common formats
            parsed_date = datetime.strptime(date, "%d/%m/%Y")
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format: {date}. Using today's date.")
            return datetime.today().strftime("%Y-%m-%d")

# Define tools using @tool decorator
@tool
def get_company_overview(stock: str) -> str:
    """Get company overview."""
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"
    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str, top_n: int = 5) -> str:
    """Get top companies by transaction volume, date must be in YYYY-MM-DD format."""
    start_date = format_date(start_date)
    end_date = format_date(end_date)
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    return retrieve_from_endpoint(url)
    
@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> str:
    """Get daily transaction for a stock, date must be in YYYY-MM-DD format."""
    start_date = format_date(start_date)
    end_date = format_date(end_date)
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"
    return retrieve_from_endpoint(url)

tools = [get_company_overview, get_top_companies_by_tx_volume, get_daily_tx]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """Answer the following queries, being as factual and analytical 
         as you can. If you need the start and end dates but they are not 
         explicitly provided, infer from the query. Whenever you return a 
         list of names, return also the corresponding values for each name. 
         If the volume was about a single day, the start and end 
         parameter should be the same."""), 
        ("human", "{input}"), 
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Initialize LLM (Ollama)
llm = ChatOllama(model="llama3.2:latest")

# Create the agent with the LLM and tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# List of queries
queries = [
    "What are the top 3 companies by transaction volume over the last 7 days?",
    "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why.",
    "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research.",
    "What is the performance of GOTO (symbol: GOTO) since its IPO listing?",
    "If I had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90-day horizon?"
]

# Run queries and print results
for query in queries:
    if not query.strip():
        logger.warning("Skipping empty query.")
        continue

    # logger.info(f"Processing query: {query}")
    try:
        result = agent.run(query)
        print("Answer:", "\n", result, "\n\n======\n\n")
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")