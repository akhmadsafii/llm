import os
import json
import requests
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
import time

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
URL_API = os.getenv("URL_API")

def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return json.dumps(response.json())
    
@tool
def get_company_overview(stock: str) -> str:
    """Retrieve QCC themes, associated teams, and their respective periods"""
    url = f"{URL_API}/get_qcc_tema_by_team"
    return retrieve_from_endpoint(url)

@tool
def get_company_overview(stock: str) -> str:
    """Get company overview"""
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"
    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str, top_n: int = 5) -> str:
    """Get top companies by transaction volume"""
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    return retrieve_from_endpoint(url)

@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> str:
    """Get daily transaction for a stock"""
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
         parameter should be the same.""" 
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Initialize AI model
llm = ChatGroq(
    temperature=0,
    model_name="qwen-2.5-32b",
    groq_api_key=GROQ_API_KEY,
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)  # Disable verbose

# Interactive mode
print("Stock AI Chatbot - Type 'exit' to quit")
while True:
    user_query = input("\nEnter questions: ")

    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Mulai hitung waktu
    start_time = time.time()

    # Mendapatkan hasil AI
    response = agent_executor.invoke({"input": user_query})

    # Akhiri hitung waktu
    end_time = time.time()
    execution_time = end_time - start_time

    # Menampilkan hasil secara langsung
    print("\nAI Response:")
    print(response["output"], flush=True)  # Print only output without verbose info
    print(f"\nExecution Time: {execution_time:.2f} seconds", flush=True)
