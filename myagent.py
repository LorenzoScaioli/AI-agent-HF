# source: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/ 
#         https://huggingface.co/learn/agents-course/unit2/langgraph/introduction

import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Import custom tools
from mytools import calculator, wolfram_query
from mytools import (
    web_page_text_extractor,
    web_search,
    wiki_search,
)

# Load environment variables
load_dotenv()

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# Aggregate all tools
TOOLS: List = [
    calculator,
    wolfram_query,
    wiki_search,
    web_search,
    web_page_text_extractor,
]

# Define system prompt
with open("system_prompt.txt", encoding="utf-8") as fp:
    SYSTEM_PROMPT = fp.read().strip()
SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)


def my_agent(model: str = "anthropic/claude-3.7-sonnet") -> StateGraph:
    """
    Define the agent workflow using LangGraph and OpenRouter.

    Args:
        model: the LLM to use.

    Returns:
        The StateGraph of the model.

    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
        )

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
    ).bind_tools(TOOLS)

    def assistant_node(state: MessagesState) -> dict:
        """
        Assistant node: add to the prompt system message and previous messages, then invokes the LLM.
        """
        history = state["messages"]
        messages = [SYSTEM_MESSAGE, *history]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Build the Langchain Graph
    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()

# if __name__ == "__main__":
#     # Example usage
#     agent = my_agent()
#     question = (
#        "If a person is 12 years younger than the French president, Emmanuel Macron, which age has this person? Search on the web for the current age of Emmanuel Macron before answering. "
#             )

#     input_message = {"messages": [HumanMessage(content=question)]}

#     response = agent.invoke(input_message)
#     used_tools = set()
#     for msg in response.get("messages", []):
#         print(f"{msg.type}: {msg.content}\n")
#         for line in msg.content.splitlines():
#             if line.startswith("[Tool:"):
#                 tool_name = line.split("]")[0].split(":", 1)[1].strip()
#                 used_tools.add(tool_name)

#     if used_tools:
#         print("Tools used during this session:", ", ".join(sorted(used_tools)))
