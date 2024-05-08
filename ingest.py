from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

class DocumentProcessor:
    def __init__(self, document_path, openai_api_key, index_name):
        load_dotenv()
        self.document_path = document_path
        self.openai_api_key = openai_api_key
        self.index_name = index_name
        print('Initialized DocumentProcessor with API Key:', self.openai_api_key)

    def load_document(self):
        loader = TextLoader(self.document_path)
        document = loader.load()
        print("Document loaded")
        return document

    def split_document(self, document):
        print("Splitting document...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)
        print(f"Created {len(texts)} chunks")
        return texts

    def embed_texts(self, texts):
        print('Generating embeddings...')
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        return embeddings.get_embeddings(texts)

    def store_vectors(self, texts, embeddings):
        print('Storing vectors to Pinecone...')
        PineconeVectorStore.from_documents(texts, embeddings, index_name=self.index_name)
        print('Ingestion completed')

    def process_document(self):
        document = self.load_document()
        texts = self.split_document(document)
        embeddings = self.embed_texts(texts)
        self.store_vectors(texts, embeddings)

if __name__ == "__main__":
    document_path = '/Users/varunchillara/git/medium-post-index/blog1.txt'
    openai_api_key = os.getenv("OPENAI_API_KEY")
    index_name = os.getenv("INDEX_NAME")
    
    processor = DocumentProcessor(document_path, openai_api_key, index_name)
    processor.process_document()
