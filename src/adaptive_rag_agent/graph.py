import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_milvus import Milvus
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from adaptive_rag_agent.models import embeddings

load_dotenv()


class RouteQuery(BaseModel):
    """
    Route query to most relevant data sources
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., description="The data source to route the query to"
    )


class GradeDocument(BaseModel):
    """
    Grade the relevance of a document to a question
    """

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="The relevance of the document to the question, either yes or no",
    )


class GradeHallucinations(BaseModel):
    """
    Grade the hallucinations in a document
    """

    binary_score: Literal["yes", "no"] = Field(
        ..., description="Whether the document contains hallucinations"
    )


class GradeAnswer(BaseModel):
    """
    Grade the answer to a question
    """

    binary_score: Literal["yes", "no"] = Field(
        ..., description="Whether the answer addresses the question, either yes or no"
    )


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

structured_llm_router = llm.with_structured_output(RouteQuery)

structured_llm_retrieval_grader = llm.with_structured_output(GradeDocument)

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)


# Prompt
router_system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents that are relevant to specific company information such as workflows, processes, policies, products, etc.
If the question can be answered using your general knowledge, use web-search."""
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_system),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)

retrieval_grader_system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question and context."""
retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_system),
        (
            "human",
            "Retrieved Document: {document}\n\nContext: {context}\n\nQuestion: {question}",
        ),
    ]
)

hallucination_grader_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

answer_grader_system = """You are a grader assessing whether an answer addresses / resolves a question and context \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system),
        (
            "human",
            "Context: \n\n {context} \n\n User question: \n\n {question} \n\n LLM generation: {generation}",
        ),
    ]
)

rewriter_system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewriter_system),
        (
            "human",
            "Context: {context}\n\nQuestion: {question}\n\nFormulate an improved question.",
        ),
    ]
)


question_router = router_prompt | structured_llm_router
retrieval_grader = retrieval_grader_prompt | structured_llm_retrieval_grader
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader
answer_grader = answer_prompt | structured_llm_answer_grader
question_rewriter = re_write_prompt | llm | StrOutputParser()

rag_system = """
    You are a helpful assistant that can answer questions based on the provided context."""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system),
        ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)

rag_chain = rag_prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(max_results=3)

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="thb",
    connection_args={"uri": "./milvus_demo.db"},
)

retriever = vector_store.as_retriever()


def format_documents(documents: List[Document]) -> List[str]:
    """Format the documents"""
    return [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]


def message_history_context_formatter(summary, messages, document_context=None):
    """Format the message history and context"""
    if summary:
        conversation_summary = (
            f"Here is a summary of the earlier conversation history: {summary}"
        )
        messages = [SystemMessage(content=conversation_summary)] + messages

    message_history = "Here is the conversation history:\n\n" + "\n".join(
        [str(message) for message in messages]
    )

    if document_context:
        document_context = "Here are the documents retrieved:\n\n" + "\n\n".join(
            document_context
        )
        return message_history + "\n\n" + document_context

    return message_history


class GraphState(MessagesState):
    question: str
    generation: str
    documents: List[str]
    summary: str
    max_iterations: int = 3
    max_generation_attempts: int = 3


def vectorstore_retrieval(state: GraphState) -> GraphState:
    """Retrieve documents from the vectorstore"""
    print("---RETRIEVE DOCUMENTS---")
    question = state.get("question", "")
    documents = retriever.invoke(question)
    documents = format_documents(documents)
    return {"documents": documents, "question": question}


def generate_answer(state: GraphState) -> GraphState:
    """Generate an answer to the question"""
    print("---GENERATE ANSWER---")
    question = state.get("question", "")
    documents = state.get("documents", [])
    summary = state.get("summary", "")
    message_history = state.get("messages", [])
    context = message_history_context_formatter(summary, message_history, documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    return {
        "generation": generation,
        "question": question,
        "documents": documents,
        "messages": [AIMessage(content=generation)],
    }


def grade_documents(state: GraphState) -> GraphState:
    """Grade the documents"""
    print("---GRADE DOCUMENTS---")
    question = state.get("question", "")
    documents = state.get("documents", [])
    summary = state.get("summary", "")
    message_history = state.get("messages", [])
    context = message_history_context_formatter(summary, message_history)
    filtered_documents = []
    for document in documents:
        grade = retrieval_grader.invoke(
            {"question": question, "document": document, "context": context}
        )
        if grade.binary_score == "yes":
            print(f"---Document {document} is relevant to the question---")
            filtered_documents.append(document)
        else:
            print(f"---Document {document} is not relevant to the question---")

    return {"documents": filtered_documents, "question": question}


def transform_query(state: GraphState) -> GraphState:
    """Transform the question"""
    print("---TRANSFORM QUERY---")
    question = state.get("question", "")
    documents = state.get("documents", [])
    summary = state.get("summary", "")
    message_history = state.get("messages", [])
    context = message_history_context_formatter(summary, message_history)
    transformed_question = question_rewriter.invoke(
        {"question": question, "context": context}
    )
    return {
        "question": transformed_question,
        "documents": documents,
        "messages": [HumanMessage(content=transformed_question)],
    }


def web_search(state: GraphState) -> GraphState:
    """Web search"""
    print("---WEB SEARCH---")
    question = state.get("question", "")
    documents = state.get("documents", [])
    web_search_results = web_search_tool.invoke({"query": question})
    web_search_results = "\n".join([result["content"] for result in web_search_results])
    return {"documents": documents + [web_search_results], "question": question}


def summarise_conversation(state: GraphState) -> GraphState:
    """Summarise the conversation"""
    messages = state.get("messages", [])
    if len(messages) <= 3:
        return state

    print("---SUMMARISE CONVERSATION---")

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

    delete_messages = [
        RemoveMessage(id=message.id) for message in state["messages"][:-2]
    ]
    return {"summary": summary.content, "messages": delete_messages, "documents": []}


## EDGES


def route_question(state: GraphState) -> GraphState:
    """Route the question to a vectorstore or web search"""
    print("---ROUTE QUESTION---")
    question = state.get("question", "")
    summary = state.get("summary", "")
    message_history = state.get("messages", [])
    context = message_history_context_formatter(summary, message_history)
    route_query = question_router.invoke({"question": question, "context": context})
    if route_query.datasource == "vectorstore":
        return "vectorstore"
    elif route_query.datasource == "websearch":
        return "websearch"


def decide_to_generate_answer(state: GraphState) -> GraphState:
    """Decide to generate an answer"""
    print("---ASSESS GRADED DOCUMENTS---")
    documents = state.get("documents", [])
    max_iterations = state.get("max_iterations", 3) - 1

    if not documents:
        print("---DECISION: ALL DOCUMENTS NOT RELEVANT. TRANSFORM QUERY---")
        if max_iterations > 0:
            return "transform_query"
        else:
            state["generation"] = (
                "No documents were relevant to the question. Please try again."
            )
            state["max_iterations"] = 3
            return "no_documents"
    else:
        print("---DECISION: SOME DOCUMENTS RELEVANT. GENERATE ANSWER---")
        return "generate_answer"


def grade_generated_answer(state: GraphState) -> GraphState:
    """Grade the generated answer"""
    print("---GRADE GENERATED ANSWER---")
    question = state.get("question", "")
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    hallucination_grade = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade.binary_score == "yes":
        print("---DECISION: ANSWER GROUNDED IN DOCUMENTS. ASSESS ANSWER---")
        summary = state.get("summary", "")
        message_history = state.get("messages", [])
        context = message_history_context_formatter(summary, message_history)
        answer_grade = answer_grader.invoke(
            {"question": question, "generation": generation, "context": context}
        )
        if answer_grade.binary_score == "yes":
            print("---DECISION: ANSWER ADDRESSES QUESTION--")
            return "finished"
        else:
            print("---DECISION: ANSWER DOES NOT ADDRESS QUESTION---")
            return "not_useful"
    else:
        print("---DECISION: ANSWER IS NOT GROUNDED IN DOCUMENTS. RETRY---")
        if state.get("max_generation_attempts", 3) > 0:
            state["max_generation_attempts"] = (
                state.get("max_generation_attempts", 3) - 1
            )
            return "not_supported"
        else:
            state["generation"] = (
                "Could not generate a supported answer. Please try again."
            )
            state["max_generation_attempts"] = 3
            return "finished"


def should_summarise(state: GraphState) -> GraphState:
    """Decide whether to summarise the conversation"""
    messages = state.get("messages", [])
    if len(messages) > 3:
        print("---DECISION: SUMMARISE CONVERSATION---")
        return "summarise_conversation"
    return END


workflow = StateGraph(GraphState)

workflow.add_node("transform_query", transform_query)
workflow.add_node("vectorstore_retrieval", vectorstore_retrieval)
workflow.add_node("web_search", web_search)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("summarise_conversation", summarise_conversation)

workflow.add_edge(START, "transform_query")
# go to web_search or vectorstore_retrieval
workflow.add_conditional_edges(
    "transform_query",
    route_question,
    {"vectorstore": "vectorstore_retrieval", "websearch": "web_search"},
)

workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("vectorstore_retrieval", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate_answer,
    {
        "generate_answer": "generate_answer",
        "transform_query": "transform_query",
        "no_documents": "summarise_conversation",
    },
)

workflow.add_conditional_edges(
    "generate_answer",
    grade_generated_answer,
    {
        "finished": "summarise_conversation",
        "not_supported": "generate_answer",
        "not_useful": "transform_query",
    },
)

workflow.add_edge("summarise_conversation", END)

adaptive_rag_agent = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=[
        "transform_query",
        "web_search",
        "generate_answer",
        "grade_documents",
        "summarise_conversation",
    ],
)
