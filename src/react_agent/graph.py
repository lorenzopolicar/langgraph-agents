"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

import numexpr
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.tools import BaseTool, tool
from langchain_milvus import Milvus
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.state import InputState, State
from react_agent.tools import TOOLS

# Tools


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"

llm = AzureChatOpenAI(
    model_name="gpt-4o-mini",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4O_MINI"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING"),
)

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="thb",
    connection_args={"uri": "./milvus_demo.db"},
)

retriever = vector_store.as_retriever()
retriever = tool(retriever.invoke)
retriever.name = "VectorStoreRetriever"

tools = [*TOOLS, retriever, calculator]


"""Default prompts used by the agent."""

SYSTEM_PROMPT = """You are a helpful AI assistant that can answer questions about the user's data. You can use the following tools to answer questions:

{tools}

Prioritise using the vectorstore retriever to answer questions. If you can't answer the question using the retriever, then use web search.

System time: {system_time}

Previous Messages Context: {context}

"""


# Nodes
class GraphState(MessagesState):
    """State for the graph."""

    summary: str
    max_steps: int = 10


async def call_model(state: GraphState) -> GraphState:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Initialize the model with tool binding. Change the model or add more tools here.
    model = llm.bind_tools(tools)

    messages = state.get("messages", [])

    if state.get("summary"):
        conversation_summary = f"Here is a summary of the earlier conversation history: {state.get('summary')}"
        messages = [AIMessage(content=conversation_summary)] + messages

    message_history = "Here is the conversation history:\n\n" + "\n".join(
        [str(message) for message in messages]
    )

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = SYSTEM_PROMPT.format(
        system_time=datetime.now(tz=timezone.utc).isoformat(),
        context=message_history,
        tools=tools,
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.get("messages", [])]
        ),
    )
    max_steps = state.get("max_steps", 10)
    max_steps = max_steps - 1

    # Handle the case when it's the last step and the model still wants to use a tool
    if max_steps == 0 and response.tool_calls:
        max_steps = 10
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ],
            "max_steps": max_steps,
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def summarise_conversation(state: GraphState) -> GraphState:
    """Summarise the conversation"""
    messages = state.get("messages", [])
    if len(messages) <= 3:
        return state

    summary = state.get("summary", "")

    if summary:
        conversation_summary = (
            f"Here is a summary of the earlier conversation history: {summary}"
            "Extend the summary to include the latest messages."
        )
    else:
        conversation_summary = "Create a summary of the conversation history."

    messages = messages + [HumanMessage(content=conversation_summary)]
    summary = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=message.id) for message in state["messages"]]
    return {"summary": summary.content, "messages": delete_messages, "documents": []}


# Edges


def route_model_output(state: GraphState) -> Literal["tools", "summarise_conversation"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.get("messages", [])[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "summarise_conversation"
    # Otherwise we execute the requested actions
    return "tools"


# Define a new graph
builder = StateGraph(GraphState)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("summarise_conversation", summarise_conversation)

# Set the entrypoint as `call_model`
builder.add_edge(START, "call_model")

# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Add a normal edge from `summarise_conversation` to `END`
builder.add_edge("summarise_conversation", END)

# Compile the builder into an executable graph
react_agent = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],
    interrupt_after=["tools"],
)
react_agent.name = "ReAct Agent"
