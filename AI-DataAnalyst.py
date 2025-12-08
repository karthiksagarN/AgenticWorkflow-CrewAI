#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-08T22:34:28.663Z
"""

from crewai import Agent,Task,Crew,Process
from crewai_tools import SerperDevTool, Tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
GROQ_API_KEY= os.getenv("GROQ_API_KEY")
SERPER_API_KEY= os.getenv("SERPER_API_KEY")
WEATHER_API=os.getenv("WEATHER_API_KEY")

llm = ChatGroq(model="gemma-7b-it",groq_api_key=os.getenv("GROQ_API_KEY"))

# Part 1 - Create an agent in CrewAI with the role of 'Data Analyst'. The agent should have a goal of 'analyzing sales data', 
# a backstory of 'an experienced data analyst with a focus on sales trends', and a tool for 'data visualization'. 
# Include the code snippet to define this agent
from crewai_tools import tool

search_tool = SerperDevTool()

# Define a custom data visualization tool
@tool('data_visualization')
def data_visualization_tool(question: list) -> str:
    """ 
    This is data visualization tool which is used to visualize sales data and analyze the trends in the data.
    """
    response = llm.invoke(question)
    return response

data_visualization = data_visualization_tool

data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Analyze sales data",
    backstory="An experienced data analyst with a focus on sales trends",
    tools=[data_visualization, search_tool],
    verbose=True
)

# Part 2 - Write a CrewAI task to 'collect sales data' for the 'Data Analyst' agent created in the previous part. 
# The task should include a description and expected output.
sales_data_last_quarter = Task(
    description="Collect sales data of OpenAI for the Last Quarter.",
    agent=data_analyst_agent,
    expected_output="A detailed report containing the collected sales data of the last quarter year",
)


collect_sales_data_task = Task(
    description="Collect sales data for analysis",
    agent=data_analyst_agent,
    expected_output="A detailed report containing the collected sales data",
    context=[sales_data_last_quarter]
)

# Part 3 - Implement a custom tool in CrewAI that connects to a weather API and retrieves the current weather for a given location. 
# Include the code to define and register this tool.
# from crewai.tools import Tool, tool_decorator

import requests

def tool_decorator(name, description):
    def decorator(func):
        func._tool_name = name
        func._tool_description = description
        return func
    return decorator

@tool_decorator(name="WeatherTool", description="Fetches current weather for a given location")
def get_current_weather(location: str) -> str:
    API_KEY = WEATHER_API  # Replace with your actual API key
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if 'current' in data:
            weather = data['current']['condition']['text']
            temperature = data['current']['temp_c']
            return f"The current weather in {location} is {weather} with a temperature of {temperature}°C."
        else:
            return f"Error: 'current' data not found in response for {location}. Response: {data}"
    
    except requests.RequestException as e:
        return f"Error: Unable to fetch weather data for {location}. Exception: {str(e)}"

# Custom Tool class for registering tools
class Tool:
    def __init__(self, name, func):
        self.name = name
        self.func = func

# Register the tool
weather_tool = Tool(name="WeatherTool", func=get_current_weather)


# Example usage of the registered tool
print(weather_tool.func("New York"))

# Example usage of the registered tool
print(weather_tool.func("Mumbai"))

# Example usage of the registered tool
print(weather_tool.func("Hyderabad"))

# Part 4
from crewai import Process
from crewai import Crew

# Define the Manager agent
manager_agent = Agent(
    role="Manager",
    goal="Delegate tasks to appropriate agents",
    backstory="A seasoned manager responsible for overseeing data collection and analysis",
    tools=[],
    verbose=True
)

# Create and run the crew
sales_data_crew = Crew(
    agents=[manager_agent, data_analyst_agent],
    tasks=[sales_data_last_quarter, collect_sales_data_task],
    process=Process.hierarchical,
    manager_llm = llm,
    verbose=True
)

# Running the crew
result = sales_data_crew.kickoff()
print("###################################################")
print(result)

# Running the crew
result = sales_data_crew.kickoff()
print("###################################################")
print(result)

# PART 5 -Write a callback function in CrewAI that logs the output of each step in a hierarchical process to a file. 
# Include the code snippet to define and use this callback in the process.

import logging
from crewai import Process

# Set up logging
logging.basicConfig(filename='crew_ai_steps.log', level=logging.INFO)

def callbacks(name):
    def call(func):
        func._tool_name = name
        return func
    return call

class StepCallback:
    def __init__(self, func):
        self.func = func

@callbacks(name="log_step_output")
def log_step_output(step, output):
    logging.info(f"Step: {step}, Output: {output}")

# Define the callback
step_callback = StepCallback(func=log_step_output)

# Create and run the crew with the callback
sales_data_crew_with_callback = Crew(
    agents=[data_analyst_agent],
    tasks=[collect_sales_data_task],
    process=Process.hierarchical,
    manager_llm = llm,
    manager_agent= manager_agent,
    step_callback=step_callback,
    verbose=True
)

# Execute the crew
result_with_callback = sales_data_crew_with_callback.kickoff()
print(result_with_callback)

# PART 6 - Write a function to create a list of agents with different roles (e.g., 'Researcher', 'Writer', 'Editor') and assign them unique goals and 
# backstories. Implement this function in CrewAI.

from crewai import Agent

def create_agents():
    roles = ["Researcher", "Writer", "Editor"]
    goals = ["Conduct research on given topics", "Write content based on research", "Edit and proofread the content"]
    backstories = [
        "A researcher with extensive experience in academic research",
        "A writer skilled in creating engaging and informative content",
        "An editor with a keen eye for detail and grammar"
    ]
    agents = []
    for role, goal, backstory in zip(roles, goals, backstories):
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=[],
            verbose=True
        )
        agents.append(agent)
    return agents

agents = create_agents()
for agent in agents:
    print(agent)

# PART 7 - Implement a CrewAI process that alternates between two agents, 'Content Creator' and 'Proofreader', to create and review articles sequentially. 
# Define the agents, tasks, and process.

content_creator_agent = Agent(
    role="Content Creator",
    goal="Create high-quality articles",
    backstory="A skilled content creator with a knack for engaging storytelling",
    tools=[],
    verbose=True
)

proofreader_agent = Agent(
    role="Proofreader",
    goal="Review and proofread articles for errors",
    backstory="An experienced proofreader with an eye for detail",
    tools=[],
    verbose=True
)

create_article_task = Task(
    description="Create an article on the given topic: The impact of AI on modern industries",
    agent=content_creator_agent,
    expected_output="A well-written article on the specified topic",
    context=[]
)

review_article_task = Task(
    description="Review the created article for any grammatical and factual errors",
    agent=proofreader_agent,
    expected_output="A proofread version of the article with corrections highlighted",
    context=[create_article_task]
)

# sequential_process = HierarchicalProcess(
#     manager=content_creator_agent,
#     agents=[content_creator_agent, proofreader_agent],
#     tasks=[create_article_task, review_article_task]
# )

content_creation_crew = Crew(
    agents=[content_creator_agent, proofreader_agent],
    tasks=[create_article_task, review_article_task],
    process=Process.sequential,
    verbose=True
)

result = content_creation_crew.kickoff()
print(result)