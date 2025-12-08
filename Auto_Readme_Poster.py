#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: Auto_Assignment_Grader.ipynb
Conversion Date: 2025-12-08T22:34:59.157Z
"""

from github import Github
from dotenv import load_dotenv
import os
import openai

load_dotenv()

# Replace with your personal access token
access_token = os.getenv("GIT_ACCESS_TOKEN")

# Public repository to analyze
repo_name = "dair-ai/ML-Papers-of-the-Week"

# Initialize the Github object
g = Github(access_token)

try:
    # Print user information to confirm token validity
    user = g.get_user()
    print(f"Authenticated as: {user.login}")

    # Get the repository
    repo = g.get_repo(repo_name)
    print(f"Successfully accessed repository: {repo.full_name}")

    # Get the last commit
    last_commit = repo.get_commits()[0]
    print(f"Last commit SHA: {last_commit.sha}")

    # Get the files changed in the last commit
    files = last_commit.files

    # Find the changes in README.md
    readme_changes = None
    for file in files:
        if file.filename == "README.md":
            readme_changes = file.patch
            break

    if readme_changes:
        # Save the changes to a local file
        with open("README_changes.patch", "w", encoding="utf-8") as file:
            file.write(readme_changes)
        print("Changes in README.md downloaded successfully.")
    else:
        print("No changes in README.md in the last commit.")

except Exception as e:
    print(f"Error: {e}")


api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = ""
api_version = "2024-02-15-preview"

with open('README_changes.patch', 'r', encoding='utf-8') as file:
    patch_content = file.read()


def get_completion(prompt):
  client = openai.AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version="api_version",
    )

  messages = [{"role": "user", "content": prompt}]
  res = client.chat.completions.create(
        model="gpt4-32k-njc",
        messages=messages,
        temperature=0
    )
  return res.choices[0].message.content


# Define a prompt to extract AI/ML papers from the patch content
prompt = f"""
Extract the list of AI/ML papers from the following patch content. Give response in json format in which all attributes of a paper are
single element of array. Paper Title, Paper Description, Paper Link, Tweet Link should be attribute of each element of the array.

Patch Content:

{patch_content}

Provide the output in the format: Paper Title, Paper Link, Tweet Link.
"""

ans = get_completion(prompt)
print(ans)

import requests
import json
import os

# Sample JSON structure containing the papers information
papers = json.loads(ans)


# Create a directory to save the downloaded papers
os.makedirs("papers", exist_ok=True)

# Function to download a file from a given URL
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {filename}")

# Download each paper
for paper in papers:
    paper_title = paper["Paper Title"]
    paper_link = paper["Paper Link"]
    # Create a safe filename by replacing spaces with underscores and removing problematic characters
    filename = f"papers/{paper_title.replace(' ', '_').replace('/', '_')}.pdf"
    download_file(paper_link, filename)






import os
import fitz  # PyMuPDF

# Directory containing the downloaded PDFs
pdf_directory = "papers"

# Directory to save the extracted text files
text_directory = "extracted_texts"
os.makedirs(text_directory, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"
    return text

# Extract text from each PDF and save to a .txt file
for pdf_filename in os.listdir(pdf_directory):
    if pdf_filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_filename)
        try:
            text = extract_text_from_pdf(pdf_path)

            # Save the extracted text to a .txt file
            text_filename = os.path.splitext(pdf_filename)[0] + ".txt"
            text_path = os.path.join(text_directory, text_filename)
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)

            print(f"Extracted text from {pdf_filename} and saved to {text_filename}")
        except Exception as e:
            print(f"Failed to extract text from {pdf_filename}: {e}")