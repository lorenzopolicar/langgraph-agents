import operator
import os
import re
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()


def download_db():
    db = SQLDatabase.from_uri("mariadb+pymysql://userconnect@10.1.93.4/cms")
    return db


db = download_db()
mini_llm = AzureChatOpenAI(
    model_name="gpt-4o-mini",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4O_MINI"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

llm = AzureChatOpenAI(
    model_name="gpt-4o",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4O"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_4O"),
    temperature=0,
)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


class State(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    sufficient_info: bool
    information: str
    query_tasks: list[str]
    sql_queries: Annotated[list, operator.add]
    query: str
    max_retries: int = 3
    answer: str


class QueryTask(TypedDict):
    question: str
    task: str
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    information: str


class Subtasks(BaseModel):
    subtasks: list[str] = Field(
        description="The list of subtasks to be transformed into queries"
    )


class Query(BaseModel):
    query: str = Field(description="The SQL query to be executed")
    reasoning: str = Field(description="The reasoning for the query")


class SufficientTables(BaseModel):
    sufficient: bool = Field(
        description="Whether the information from the tables are sufficient to answer the user question"
    )
    reason: str = Field(description="The reason for the answer")


class TransformUserQuestion(BaseModel):
    question: str = Field(description="The transformed user question")


transform_user_question_system = """

You will receive a user question and your job is to make the question more specific and clear.
Try to make the question as simple as possible.

"""

transform_user_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", transform_user_question_system),
        ("placeholder", "{messages}"),
    ]
)

transform_user_question_llm = mini_llm.with_structured_output(TransformUserQuestion)

selector_system = """
You are provided with a user's natural language query alongside a list of table names and their corresponding descriptions. 
Your task is to identify all tables that could potentially contribute to answering the query.
You could also receive feedback from your last iteration of selecting tables. Take the feedback into account. In this case, select tables that you have not selected in the previous iteration.

Instructions:

Inclusiveness is Key:
Err on the side of inclusion. If a table might be relevant—even indirectly—include it. When in doubt, select the table.

Analytical Review: 
Examine each table's name and description carefully.
Consider potential joins or relationships: a table that seems marginal on its own may become crucial when combined with others.

Evaluate Relevance:
Identify keywords, data types, or any hints in the descriptions that suggest the table might hold useful data.
Look for connections between tables (e.g., shared fields or related concepts) that might allow for joining to gather comprehensive data.

Call the get_schema_tool to get the schema of these selected tables.
"""

selector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", selector_system),
        ("placeholder", "{messages}"),
    ]
)

contextualiser_system = """
You are a specialist at organising and structuring information and you have strong attention to detail.

You will receive the following information:
- A user question
- A list of all tables and their descriptions in the database 
- A list of selected relevant tables and their schemas. The schemas have a comment section that describes the table and the relationship with other tables.

Your job is to organise the information from the selected tables and their schemas into a structured format and to eliminate fields in the schemas that are not relevant to the user question.
When in doubt, keep the field. Retain the COMMENT section of the schema including all the joins information which is crucial. You can make this it's own section. Ensure that everything is detailed and clear and do not cut too much information.
You should not try to answer the user question, just present the information in a clear and structured format.

Output Format:
- User Question: <The user question>
- Tables and Schemas: 1. A description of the table. 2. The schema of the table. 3. The COMMENT section of the schema.
- Key Relationships: <A list of key relationships between the tables including the join conditions>
- Relevant Fields for the User Question: <A list of relevant fields for the user question>
"""

contextualiser_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualiser_system),
        ("placeholder", "{messages}"),
    ]
)

sufficient_tables_system = """ 
You are an experienced and professional database administrator.

You receive the schemas (including detailed table comments) of the tables selected by the Selector LLM along with the original user query. 
Your job is to decide whether these tables are sufficient to generate an SQL query that can be executed to answer the user's query.
If they are not sufficient, you must provide a clear explanation of what is missing and suggest additional tables.

Instructions:

Deep Schema and Comment Analysis:
Carefully read the table COMMENTs, as they explain the information each table holds and describe relationships and join information with other tables.
Examine the columns, keys (primary, foreign), and any other structural details in the schemas that is detailed.

Identify Missing Tables from Comments:
If the table COMMENTs mention any table names that were not already selected (i.e. the table is not in the list of schemas), consider those as necessary for answering the query.
In such cases, output that the current selection is insufficient and explicitly list the missing table names as suggestions for the reason, indicating that they should be routed back to the Selector LLM for inclusion.

Sufficiency Evaluation:
Decide if the available tables, when appropriately joined together, can be used to generate an sql query that can be executed to answer the user's query.
If this is the case, state that the selection is sufficient. When in doubt, just mark it as sufficient.
If any critical tables (including those mentioned in comments) are missing, mark the decision as insufficient.

Output Format:
Sufficient: Return a boolean value (True or False) indicating whether the tables are sufficient. Return True if the tables are sufficient.
Reason: Provide a detailed explanation of your reasoning. If the tables are insufficient, suggest any additional tables that might be relevant to the user query.
"""

sufficient_tables_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sufficient_tables_system),
        ("placeholder", "{messages}"),
    ]
)

sufficient_tables_llm = llm.with_structured_output(SufficientTables)

query_gen_system = """
You are provided with a user question, a list of table names, schema summaries, join conditions, and any additional notes necessary for constructing a MariaDB SQL query. 
Additionally, you may sometimes receive an error message indicating that the previously generated query failed upon execution. 

Instructions:

If you don't get an error message, generate an SQL query that is syntactically correct.

Initial Query Generation:
Analyze the Context: Use the provided context to understand the tables involved, their schemas, join conditions, and any filtering or grouping requirements needed to answer the user query.
Construct the SQL Query: Generate a valid MariaDB SQL query that incorporates all necessary joins, filters, and calculations. Joins must use indexed fields.
Include Inline Comments: Annotate the query to explain how various parts of the provided context are being utilized (e.g., why specific joins or filters are applied).
Output Format: Return a single SQL statement. 

If you get an error message, generate a new SQL query that is syntactically correct.

Error Correction:
Review the Error: If an error message is provided along with the context, analyze the error to determine the underlying issue (e.g., syntax errors, aliasing problems, missing join conditions).
Modify the Query: Adjust the previously generated SQL query to correct the error while ensuring that it still meets the requirements derived from the context.
Document the Correction: Include brief inline comments in the query to describe the changes made to address the error.
Output Format: Return the corrected SQL query as a single statement. 

Double check the MariaDB query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins
"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_gen_system),
        ("placeholder", "{messages}"),
    ]
)

query_gen_llm = llm.with_structured_output(Query)

decomposer_system = """
You are an expert SQL developer.

You will receive a user question that represents a query to a database.

If the user question is simple and doesn't require any breakdown, return a list with a single task (original user question).

If the user question is complex and has multiple subqueries, decompose the user question into a list of subtasks that can be executed in parallel.

Each subtask should be a well defined subtask that can be transformed into an independent SQL query.

If in doubt, return a list with a single task (original user question).
"""

decomposer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", decomposer_system),
        ("placeholder", "{messages}"),
    ]
)

decomposer_llm = llm.with_structured_output(Subtasks)

reducer_system = """
You are an expert SQL developer.

You will receive a list of SQL queries and a user question.

These SQL queries are subtasks that are part of the overall task to answer the user question.

Your job is to reduce the list of MariaDB SQL queries into a single MariaDB SQL query that answers the user question.

Try to combine the queries into a single query if possible and make sure that the query is efficient.

The SQL query should be syntactically correct.
"""

reducer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", reducer_system),
        ("placeholder", "{messages}"),
    ]
)

reducer_llm = llm.with_structured_output(Query)

answer_and_reasoning_system = """
You are an advanced reasoning agent.

You will receive a user question, some intermediate reasoning to generate an sql query, and the final raw output from an SQL query. 

The SQL agent follows the process of decomposing the user question into a list of subtasks,
transforming the subtasks into SQL queries, checking the queries, reducing the queries into a single query,
executing the query, iteratively refining the query until it is correct, and finally returning the final answer.

Your job is to answer the user question by interpreting the information provided in the messages and present it in a structured and detailed manner.

Provide a structured chain of thought reasoning for your answer. Go through the reasoning step by step and the intermediate results.

Your answer should be in the same language as the user question.

If you don't have enough information to answer the question, say "I don't know".
"""

answer_and_reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_and_reasoning_system),
        ("placeholder", "{messages}"),
    ]
)


# Nodes
def first_tool_call(state: State) -> State:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


def info_sql_database_tool_call(state: State) -> State:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "info_sql_database_tool", "args": {}, "id": "tool_abcd123"}
                ],
            ),
        ]
    }


def handle_tool_error(state: State) -> State:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, State]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


@tool
def db_query_tool(query: str) -> str:
    """
    Query the database with a MariaDB SQL query
    """
    result = db.run_no_throw(query)
    if result.startswith("Error:"):
        return (
            "Error: Query failed. Please rewrite your query and try again."
            + f"\n\nDetails:{result}"
        )
    if len(result) > 2000:
        return result[:2000] + "..."
    return result


@tool
def info_sql_database_tool() -> str:
    """
    Get the information about the tables in the database
    """
    result = db.run_no_throw(
        "SELECT TABLE_NAME, TABLE_COMMENT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'cms' AND LENGTH(TABLE_COMMENT) > 100;"
    )
    full_response = f"Tables and Descriptions:\n\n{result}"
    return full_response


@tool
def get_schema_tool(table_name: str) -> str:
    """
    get the schema of the table
    """
    schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    result = schema_tool.invoke(table_name)
    pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
    result = re.sub(pattern, "", result)
    result = f"Selected Table {table_name}\n\nSchema: {result}"
    return result


def transform_user_question(state: State) -> State:
    prompt = transform_user_question_prompt.format(messages=state["messages"])
    response = transform_user_question_llm.invoke(prompt)
    return {
        "question": response.question,
        "messages": [AIMessage(content=response.question)],
    }


def selector(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    messages = [user_question] + [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]
    last_message = state["messages"][-1]
    max_retries = state["max_retries"]
    sufficient_info = state.get("sufficient_info", True)
    if not sufficient_info:
        max_retries -= 1
        messages.append(last_message)
    prompt = selector_prompt.format(messages=messages)
    print(prompt)
    model_get_schema = mini_llm.bind_tools([get_schema_tool], tool_choice="required")
    response = model_get_schema.invoke(prompt)
    return {"messages": [response], "max_retries": max_retries}


def contextualiser(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    # GET LAST 6 TOOL MESSAGES
    messages = [user_question] + [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ][-6:]
    prompt = contextualiser_prompt.format(messages=messages)
    print(prompt)
    response = llm.invoke(prompt)
    return {"messages": [response], "information": response.content}


def sufficient_tables(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information]
    prompt = sufficient_tables_prompt.format(messages=messages)
    print(prompt)
    response = sufficient_tables_llm.invoke(prompt)
    full_response = (
        f"Sufficient Tables: {response.sufficient}\n\nReason: {response.reason}"
    )
    return {
        "messages": [AIMessage(content=full_response)],
        "sufficient_info": response.sufficient,
    }


def decomposer(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information]
    prompt = decomposer_prompt.format(messages=messages)
    response = decomposer_llm.invoke(prompt)
    full_response = f"Subtasks: {response.subtasks}"
    return {
        "query_tasks": response.subtasks,
        "messages": [AIMessage(content=full_response)],
    }


def query_gen(state: QueryTask) -> State:
    last_message = state["messages"][-1]

    task = AIMessage(content=f"Task: {state['task']}")
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information, task]

    if last_message.content.startswith("Error:"):
        messages.append(last_message)

    prompt = query_gen_prompt.format(messages=messages)
    print(prompt)
    response = query_gen_llm.invoke(prompt)

    full_response = f"Query: {response.query}\n\nReasoning: {response.reasoning}"
    return {
        "messages": [AIMessage(content=full_response)],
        "sql_queries": [response.query],
    }


def reducer(state: State) -> State:
    subtasks = AIMessage(content="\n".join(state["query_tasks"]))
    sql_queries = AIMessage(content="\n".join(state["sql_queries"]))
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information, subtasks, sql_queries]
    prompt = reducer_prompt.format(messages=messages)
    print(prompt)
    response = reducer_llm.invoke(prompt)
    full_response = f"Query: {response.query}\n\nReasoning: {response.reasoning}"
    state["query"] = response.query
    return {"query": response.query, "messages": [AIMessage(content=full_response)]}


def get_query_execution(state: State) -> State:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "db_query_tool",
                        "args": {"query": state["query"]},
                        "id": "query_execution",
                    },
                ],
            )
        ]
    }


def final_answer(state: State) -> State:
    messages = [
        HumanMessage(content=state["question"]),
        AIMessage(content=state["information"]),
        state["messages"][-1],
    ]
    prompt = answer_and_reasoning_prompt.format(messages=messages)
    print(prompt)
    response = mini_llm.invoke(prompt)
    return {
        "answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }


# Edges
def map_query_task(state: State):
    messages = state["messages"]
    return [
        Send(
            "query_gen",
            {
                "task": task,
                "messages": messages,
                "question": state["question"],
                "information": state["information"],
            },
        )
        for task in state["query_tasks"]
    ]


def continue_sufficient_tables(state: State) -> State:
    messages = state["messages"]
    max_retries = state["max_retries"]
    last_message = messages[-1]
    if last_message.content.startswith("Sufficient Tables: True") or max_retries == 0:
        return "decomposer"
    else:
        return "selector"


def continue_query_gen(state: State) -> State:
    messages = state["messages"]
    last_message = messages[-1]
    max_retries = state["max_retries"] - 1
    state["max_retries"] = max_retries
    if max_retries < 0:
        return "failed"
    if last_message.content.startswith("Error:"):
        task = "Fix the error in the query and rewrite the query."
        state["query_tasks"] = [RemoveMessage(id=message.id) for message in messages]
        state["query"] = ""
        return Send(
            "query_gen",
            {
                "task": task,
                "messages": messages,
                "question": state["question"],
                "information": state["information"],
            },
        )
    else:
        return "correct_query"


# Build the workflow
workflow = StateGraph(State)
workflow.add_node("transform_user_question", transform_user_question)
workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("info_sql_database_tool_call", info_sql_database_tool_call)
workflow.add_node(
    "info_sql_database_tool", create_tool_node_with_fallback([info_sql_database_tool])
)
workflow.add_node("tables_selector", selector)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
workflow.add_node("contextualiser", contextualiser)
workflow.add_node("sufficient_tables", sufficient_tables)
workflow.add_node("decomposer", decomposer)
workflow.add_node("query_gen", query_gen)
workflow.add_node("reducer", reducer)
workflow.add_node("get_query_execution", get_query_execution)
workflow.add_node("query_execute", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("final_answer", final_answer)

workflow.add_edge(START, "transform_user_question")
workflow.add_edge("transform_user_question", "info_sql_database_tool_call")
workflow.add_edge("info_sql_database_tool_call", "info_sql_database_tool")
workflow.add_edge("info_sql_database_tool", "tables_selector")
workflow.add_edge("tables_selector", "get_schema_tool")
workflow.add_edge("get_schema_tool", "contextualiser")
workflow.add_edge("contextualiser", "sufficient_tables")
workflow.add_conditional_edges(
    "sufficient_tables",
    continue_sufficient_tables,
    {
        "selector": "tables_selector",
        "decomposer": "decomposer",
    },
)
workflow.add_conditional_edges("decomposer", map_query_task, ["query_gen"])
workflow.add_edge("query_gen", "reducer")
workflow.add_edge("reducer", "get_query_execution")
workflow.add_edge("get_query_execution", "query_execute")
workflow.add_conditional_edges(
    "query_execute",
    continue_query_gen,
    {
        "query_gen": "query_gen",
        "correct_query": "final_answer",
        "failed": END,
    },
)
workflow.add_edge("final_answer", END)
sql_agent = workflow.compile()
