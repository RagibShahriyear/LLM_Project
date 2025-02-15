from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor, 
    create_react_agent,
)

from langchain_core.tools import Tool
from langchain_groq import ChatGroq

# load environment variables from .env file
load_dotenv()

# Define a very simple tool function that return the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime
    
    now = datetime.datetime.now() # Get current time
    return now.strftime("%I:%M %p") # Format time in H:MM AM/PM format

# List of tools available to the agent 
tools = [
    Tool(
        name = "Time",# Name of the tool
        func = get_current_time, # Function that the tool will execute
        # Description of the tool
        description = "Useful for when you need to know the current time",
    ),
]

# Pull the prompt template from the hub
# ReAct =  Reason and Acttion
prompt = hub.pull("hwchase17/react")

# Initialize LLM
llm = ChatGroq(model="mixtral-8x7b-32768")

# Create the react agent using create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools=tools,
    verbose=True,
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response from the agent
print("response:", response)





# # Define a simple tool function that return relevant documents based on a query
# def get_relevant_docs(*args, **kwargs):
#     """Returns relevant documents based on a query"""
#     relevant_docs = retriever.invoke(query)
#     return relevant_docs
    
# #get_relevant_docs(query=query, retriever=retriever)