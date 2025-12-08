#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: AI-DataAnalyst.ipynb
Conversion Date: 2025-12-08T22:34:54.162Z
"""

from crewai import Agent,Task,Crew,Process
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY= os.getenv("GROQ_API_KEY")
SERPER_API_KEY= os.getenv("SERPER_API_KEY")

llm = ChatGroq(model="gemma2-9b-it",groq_api_key=os.getenv("GROQ_API_KEY"))
print(llm.invoke("hey! what's up!"))
search_tool = SerperDevTool()

# Defining the assignment text and submission

assignment_text = """
Assignment: AUTOMATED FAQ ANSWERING.

Goal: 

Create a prompt that answers Frequently Asked Questions (FAQs) for a product of your choice.

Task: 

First, have the model generate examples of FAQs and corresponding answers based on a brief description of the product (maybe choose an item that you use in your everyday life). Then, use these examples as context for the model to learn from when answering a novel question about the product (you can write this novel question yourself).
"""

## Submission of a random student.
assignmentsubmission = """
Assignment 2: Automated FAQ Answering
Goal:
Create a prompt that answers Frequently Asked Questions (FAQs) for a product of your choice.
Task:
First, have the model generate examples of FAQs and corresponding answers based on a brief
description of the product (maybe choose an item that you use in your everyday life). Then, use
these examples as context for the model to learn from when answering a novel question about the
product (you can write this novel question yourself).
Firstly Lets Choose a Product and Write a prompt to the language model to
generate few FAQ's about the product and instruct the model itself to write the
answers for FAQ's.
Product Description: AI Ultra Voice Control Robot Vacuum with Matrix Clean Navigation,
Home Mapping, 60-Day Capacity, Self-Empty Base for Homes with Pets, Carpet & Hard Floors
(Silver/Black)
About this item: INCREDIBLE SUCTION: Powerful Shark suction picks up dirt and debris on
all floor types—tackling even the toughest of messes in your home.SharkClean App & Voice
Control NO SPOTS MISSED: With Matrix Clean, the robotic vacuum cleans in a precision
matrix grid taking multiple passes over dirt and debris for whole home, deep cleaning coverage.
IT EMPTIES ITSELF: The bagless, self-emptying base holds up to 60 days of dirt and debris.
The bagless base does not require any additional purchase of disposal bags like traditional auto
empty robots. SELF-CLEANING BRUSHROLL: Digs deep into carpets and directly engages
hardwood floors pulling up debris, hair, and dirt from all surfaces in your home - engineered to
pick up more hair and is anti-hair wrap. PRECISION HOME MAPPING: 360° LiDAR vision
quickly and accurately maps your home so your robot can methodically clean detecting and
avoiding objects in its path, day or night, adapting to everyday changes in the home. PERFECT
FOR HOMES WITH PETS: With powerful vacuum suction, Shark's self-cleaning brushroll, and
self-empty system the robot captures pet hair, dirt, and debris with ease. HANDS-FREE
CONTROL: Set a cleaning schedule, initiate on-demand cleaning, or activate cleans all with
sound of your voice using Amazon Alexa or Google Assistant. EXTENDED RUNTIME: With
Recharge and Resume, the robot will return to the dock, recharge, and can pick up where it left
off.
ANALYZE THE PRODUCT DESCRIPTION AND GENERATE FAQ's and ANSWER THEM
YOURSELF.
RESPONSE :
FAQs for AI Ultra Voice Control Robot Vacuum
1. How powerful is the suction of the AI Ultra Voice Control Robot Vacuum?
The AI Ultra Voice Control Robot Vacuum features powerful Shark suction that efficiently picks
up dirt and debris from all floor types, including carpets and hard floors. It is designed to tackle
even the toughest messes in your home.
2. How does the Matrix Clean Navigation work?
Matrix Clean Navigation ensures thorough cleaning by moving the robot in a precision matrix
grid. This method involves multiple passes over dirt and debris, providing deep cleaning
coverage for the entire home.
3. What is the capacity of the self-emptying base, and does it require additional purchases?
The self-emptying base can hold up to 60 days of dirt and debris. It is bagless, meaning it does
not require any additional purchases of disposal bags, unlike traditional auto-empty robots.
4. How effective is the self-cleaning brushroll?
The self-cleaning brushroll is highly effective as it digs deep into carpets and directly engages
with hardwood floors to pull up debris, hair, and dirt. It is specifically engineered to pick up more
hair and is designed to prevent hair from wrapping around the brushroll.
5. How accurate is the home mapping feature?
The precision home mapping feature utilizes 360° LiDAR vision to quickly and accurately map
your home. This allows the robot to methodically clean, detecting and avoiding objects in its
path both day and night, and adapting to everyday changes in the home environment.
NOW LET's USE THE EXAMPLE FAQ's GENERATED BY THE LANGUAGE MODEL
AS ITS CONTEXT TO LEARN FROM AND ASK A NOVEL QUESTION.
PROMPT:
“””
Product Description:
This Shark AI Ultra Voice Control Robot Vacuum delivers powerful suction for deep cleaning on
all floor types, making it perfect for homes with pets. It utilizes Matrix Clean, ensuring thorough
cleaning coverage by taking multiple passes over dirt and debris. It also houses a self-emptying
base that can hold dirt for up to 60 days without needing any disposal bags. The smart robot
vacuum features a self-cleaning brushroll designed to handle more hair and avoid tangling, and
precise home mapping technology that adapts to changes and avoids objects during cleaning
using 360° LiDar vision. You can set schedules and control cleaning through Amazon Alexa or
Google Assistant for on-demand cleaning. Plus, it has a Recharge and Resume feature allowing
the robot to continue cleaning from where it left off after recharging. Ideal for carpet and hard
floors.
Here are few sample FAQ's for AI Ultra Voice Control Robot Vacuum:
1. How powerful is the suction of the AI Ultra Voice Control Robot Vacuum?
The AI Ultra Voice Control Robot Vacuum features powerful Shark suction that efficiently picks
up dirt and debris from all floor types, including carpets and hard floors. It is designed to tackle
even the toughest messes in your home.
2. How does the Matrix Clean Navigation work?
Matrix Clean Navigation ensures thorough cleaning by moving the robot in a precision matrix
grid. This method involves multiple passes over dirt and debris, providing deep cleaning
coverage for the entire home.
3. What is the capacity of the self-emptying base, and does it require additional purchases?
The self-emptying base can hold up to 60 days of dirt and debris. It is bagless, meaning it does
not require any additional purchases of disposal bags, unlike traditional auto-empty robots.
4. How effective is the self-cleaning brushroll?
The self-cleaning brushroll is highly effective as it digs deep into carpets and directly engages
with hardwood floors to pull up debris, hair, and dirt. It is specifically engineered to pick up more
hair and is designed to prevent hair from wrapping around the brushroll.
5. How accurate is the home mapping feature?
The precision home mapping feature utilizes 360° LiDAR vision to quickly and accurately map
your home. This allows the robot to methodically clean, detecting and avoiding objects in its
path both day and night, and adapting to everyday changes in the home environment.
Use this compositional context to answer a novel question about the product :
“ What if there are objects in the room which are unrecognizable by the Robot Vacuum? ”
“””
RESPONSE:
What if there are objects in the room which are unrecognizable by the Robot Vacuum?
The AI Ultra Voice Control Robot Vacuum is equipped with advanced 360° LiDAR vision,
which allows it to map your home accurately and detect most objects in its path. However, in the
event that the robot encounters unrecognizable objects, the vacuum's intelligent navigation
system is designed to adapt and avoid these obstacles to prevent any potential issues during
cleaning. The precise home mapping technology ensures that the robot can navigate around
unfamiliar items, maintaining its cleaning efficiency while avoiding collisions or getting stuck. If
the object is too small or unusual, you may want to remove it from the floor to ensure the robot
can clean seamlessly and effectively. This adaptability ensures that the vacuum can handle
dynamic home environments and provide a consistent cleaning performance.
N Karthik Sagar karthik.sagarn@gmail.com 3ALTERNATIVE PROMPT :
Alternative Prompt is because the previous prompt contains more INPUT tokens. So, using this
alternative prompt we can generate a similar response as before.
PROMPT:
“””
Using your everyday Shark AI Ultra Voice Control Robot Vacuum as a product example,
generate a set of Frequently Asked Questions (FAQs) and their corresponding answers based on
the given product description. This Shark AI Ultra Voice Control Robot Vacuum delivers
powerful suction for deep cleaning on all floor types, making it perfect for homes with pets. It
utilizes Matrix Clean, ensuring thorough cleaning coverage by taking multiple passes over dirt
and debris. It also houses a self-emptying base that can hold dirt for up to 60 days without
needing any disposal bags. The smart robot vacuum features a self-cleaning brushroll designed to
handle more hair and avoid tangling, and precise home mapping technology that adapts to
changes and avoids objects during cleaning using 360° LiDar vision. You can set schedules and
control cleaning through Amazon Alexa or Google Assistant for on-demand cleaning. Plus, it has
a Recharge and Resume feature allowing the robot to continue cleaning from where it left off
after recharging. Ideal for carpet and hard floors.
Once you generate FAQs and answers, use this compositional context to answer a novel question
about the product: “ What if there are objects in the room which are unrecognizable by the
Robot Vacuum? ”
“””
RESPONSE:
Frequently Asked Questions (FAQs)
1. What types of floors can the Shark AI Ultra Voice Control Robot Vacuum clean?
• The Shark AI Ultra Voice Control Robot Vacuum delivers powerful suction suitable for
all floor types, including carpets and hard floors.
2. Is the Shark AI Ultra Voice Control Robot Vacuum effective for homes with pets?
• Yes, it is ideal for homes with pets due to its powerful suction and self-cleaning brushroll
designed to handle more hair and avoid tangling.
3. How does the Matrix Clean technology work?
• The Matrix Clean technology ensures thorough cleaning coverage by taking multiple
passes over dirt and debris, leaving your floors cleaner.
4. How long can the self-emptying base hold dirt?
• The self-emptying base can hold dirt for up to 60 days without needing any disposal bags.
5. Can I control the Shark AI Ultra Voice Control Robot Vacuum with my voice?
• Yes, you can set schedules and control cleaning through Amazon Alexa or Google
Assistant for on-demand cleaning.
6. What is the Recharge and Resume feature?
N Karthik Sagar karthik.sagarn@gmail.com 4• The Recharge and Resume feature allows the robot to return to its dock to recharge and
then continue cleaning from where it left off.
7. How does the robot navigate and avoid obstacles?
• The robot uses precise home mapping technology with 360° LiDar vision, allowing it to
adapt to changes and avoid objects during cleaning.
8. Does the robot require frequent maintenance for hair tangling?
• No, the self-cleaning brushroll is designed to handle more hair and avoid tangling,
reducing the need for frequent maintenance.
9. Can I schedule cleaning times for the robot vacuum?
• Yes, you can set schedules for the robot vacuum using the companion app or voice
control with Amazon Alexa or Google Assistant.
10. Is the Shark AI Ultra Voice Control Robot Vacuum easy to set up?
• Yes, the robot vacuum is user-friendly and can be easily set up through the companion
app and voice control integration.
Novel Question
What if there are objects in the room which are unrecognizable by the Robot Vacuum?
• The Shark AI Ultra Voice Control Robot Vacuum is equipped with 360° LiDar vision and
precise home mapping technology to navigate and avoid obstacles. However, if there are
objects that the robot vacuum cannot recognize, it may slow down and maneuver
carefully around them. The robot is designed to adapt to changes in its environment, but
it is recommended to clear the floor of any small or unusual objects that could potentially
disrupt the cleaning process for optimal performance.
NOTE : After Comparing the Responses of both the prompts, the Original Prompt with more
Input Tokens and the Alternative Prompt with Less Input Tokens using a language model
again like ChatGPT, GPT-4, Gemini, etc. The Response received is : The two
paragraphs show approximately 90 percent similarity.
"""

clarity_agent = Agent(
    role='Clarity and Conciseness Grader',
    goal='Evaluate the clarity and conciseness of the Product Description provided in the assignment. Check for the relevance of the Product description with the assignment.',
    backstory=f"""You are an experienced educator with a keen eye for detail, specializing in evaluating the clarity and conciseness of written content.
    Assignment: {assignment_text}""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

relevance_agent = Agent(
    role='Relevance and Focus Grader',
    goal='Evaluate the relevance and focus of the assignment submission',
    backstory=f"""You are a seasoned educator with a strong background in assessing the relevance and focus of academic work.
    Assignment: {assignment_text}""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

accuracy_agent = Agent(
    role='Accuracy Grader',
    goal='Evaluate the accuracy of the information in the assignment submission',
    backstory=f"""You are an expert in evaluating the accuracy of content, ensuring that all facts and details are correctly represented.
    Assignment: {assignment_text}""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

QuestionHandling_agent = Agent(
    role="Question Handler",
    goal="Check whether the novel question relevant and plausible for the chosen product?",
    backstory=f""" You are an expert in evaluating the accuracy of content, ensuring that all facts and details are correctly represented.
    Assignment: {assignment_text}  """,
    verbose=True,
    llm=llm,
    allow_delegation=False
)

promptdesign_agent = Agent(
    role='Prompt Design Grader',
    goal='Evaluate Prompt Design and Prompt Structure whether it is clear and ambiguous.',
    backstory=f""" You are an expert in evaluating the prompt design and structure, ensuring that all facts and details are correctly represented.
    Assignment: {assignment_text} """,
    verbose=True,
    llm=llm,
    allow_delegation=False
)

originality_agent = Agent(
    role='Originality and Creativity Grader',
    goal='Evaluate whether the submissions demonstrate creativity in the choice of product and the generation of FAQs?',
    backstory=f""" You are an expert in evaluating whether the submissions demonstrate creativity in the choice of product and the generation of FAQs, ensuring that all facts and details are correctly represented.
    Assignment: {assignment_text}  """,
    verbose=True,
    llm=llm,
    allow_delegation=False
)

tone_agent = Agent(
    role='Tone and Style Grader',
    goal='Evaluate the tone and style of the assignment submission',
    backstory=f"""You have a background in literature and writing, with extensive experience in evaluating the tone and style of written work.
    Assignment: {assignment_text}""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

examples_agent = Agent(
    role='Examples and Details Grader',
    goal='Evaluate the use of examples and details in the assignment submission',
    backstory=f"""You specialize in assessing the use of examples and details in academic and professional writing.
    Assignment: {assignment_text}""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# Create tasks for each agent
task_clarity = Task(
    description=f"""Grade the clarity and conciseness of the following assignment submission on a scale of 1-10:
    {assignmentsubmission}""",
    expected_output="Score for clarity and conciseness with comments",
    agent=clarity_agent
)

task_relevance = Task(
    description=f"""Grade the relevance and focus of the following assignment submission on a scale of 1-10:
    {assignmentsubmission}""",
    expected_output="Score for relevance and focus with comments",
    agent=relevance_agent
)

task_accuracy = Task(
    description=f"""Grade the accuracy of the information in the following assignment submission on a scale of 1-20:
    {assignmentsubmission}""",
    expected_output="Score for accuracy with comments",
    agent=accuracy_agent
)

task_qa = Task(
    description=f"""Grade the novel question of the information in the following assignment submission on a scale of 1-10:
    {assignmentsubmission}""",
    expected_output="Score for novel question with comments",
    agent=accuracy_agent
)

task_promptdesign = Task(
    description=f"""Grade the prompt design and structure of the information in the following assignment submission on a scale of 1-20:
    {assignmentsubmission}""",
    expected_output="Score for prompt design and structure with comments",
    agent=accuracy_agent
)

task_originality = Task(
    description=f"""Grade the originality and creativity of the information in the following assignment submission on a scale of 1-10:
    {assignmentsubmission}""",
    expected_output="Score for originality and creativity with comments",
    agent=accuracy_agent
)

task_tone = Task(
    description=f"""Grade the tone and style of the following assignment submission on a scale of 1-10:
    {assignmentsubmission}""",
    expected_output="Score for tone and style with comments",
    agent=tone_agent
)

task_examples = Task(
    description=f"""Grade the use of examples and details in the following assignment submission on a scale of 1-10:
    {assignmentsubmission}""",
    expected_output="Score for use of examples and details with comments",
    agent=examples_agent
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[clarity_agent, relevance_agent, accuracy_agent, QuestionHandling_agent, promptdesign_agent, originality_agent, tone_agent, examples_agent],
    tasks=[task_clarity, task_relevance, task_accuracy, task_qa, task_promptdesign, task_originality, task_tone, task_examples],
    verbose=2, # You can set it to 1 or 2 to different logging levels
    process=Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("#####################################")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(result)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[clarity_agent, relevance_agent, accuracy_agent, QuestionHandling_agent, promptdesign_agent, originality_agent, tone_agent, examples_agent],
    tasks=[task_clarity, task_relevance, task_accuracy, task_qa, task_promptdesign, task_originality, task_tone, task_examples],
    verbose=1, # You can set it to 1 or 2 to different logging levels
    process=Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("#####################################")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(result)