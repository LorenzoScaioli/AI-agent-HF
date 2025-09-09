# AI Agent for HuggimgFace course

## Project Overview
This repository contains a Python-based project designed to evaluate and submit answers for [HuggingFace AI Agents course](https://huggingface.co/learn/agents-course). The project leverages the GAIA benchmark and integrates various tools to process and respond to complex queries. The application is built using Gradio for the user interface and incorporates advanced AI capabilities through the LangChain framework.

## Features
- **Agent Implementation**: The `BasicAgent` class is the core of the project, designed to process questions and provide answers using a set of predefined tools.
- **Tool Integration**: Includes tools for arithmetic operations, web searches, Wikipedia queries, and more.
- **Gradio Interface**: Provides an interactive web-based interface for running evaluations and submitting answers.
- **Logging**: Comprehensive logging to track application behavior and agent responses.

## File Descriptions
- **`app.py`**: The main application file that defines the Gradio interface and orchestrates the agent's operations.
- **`myagent.py`**: Contains the implementation of the `BasicAgent` class, which integrates tools and processes queries.
- **`mytools.py`**: Defines custom tools such as a calculator, web search, and Wikipedia search, which are used by the agent.
- **`logs.txt`**: A log file capturing the application's runtime behavior and agent responses.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.
- **`system_prompt.txt`**: Describes the system's capabilities, tools, and formatting rules for generating answers.

## Tools
The project integrates the following tools:
- **Calculator**: Performs basic arithmetic operations.
- **WolframAlpha**: Computes complex mathematical expressions.
- **Wiki Search**: Retrieves summaries of Wikipedia articles.
- **Web Search**: Conducts DuckDuckGo searches.
- **Web Page Text Extractor**: Fetches plain text from web pages.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository-directory>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Launch the application:
   ```bash
   python app.py
   ```
2. Open the Gradio interface in your browser.
3. Log in to your Hugging Face account.
4. Click "Run Evaluation & Submit All Answers" to process questions and submit answers.

## Dependencies
- Python 3.8+
- Gradio
- Requests
- Pandas
- LangChain Core
- LangChain OpenAI
- LangGraph
- BeautifulSoup4
- WolframAlpha
- Wikipedia

## Logs
The `logs.txt` file contains detailed logs of the application's execution, including agent responses and errors.

