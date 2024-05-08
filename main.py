from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os

class QueryProcessor:
    def __init__(self):
        load_dotenv()
        print('Initializing...')
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI()
        self.index_name = os.getenv("INDEX_NAME")
        self.vector_store = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)
        self.retrieve_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.retrieve_qa_chat_prompt)
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.vector_store.as_retriever(),
            combine_docs_chain=self.combine_docs_chain
        )

    def process_query(self, query):
        print('Processing query...')
        result = self.retrieval_chain.invoke(input={"input": query})
        return result

if __name__ == "__main__":
    processor = QueryProcessor()
    query = 'what exactly does pinecone do?'
    result = processor.process_query(query)
    print(result)
