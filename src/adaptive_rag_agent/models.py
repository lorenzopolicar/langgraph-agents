import os

from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING"),
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4O_MINI"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

URI = "./milvus_demo.db"
collection_name = "thb"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name=collection_name,
)
