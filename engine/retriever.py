import json
import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


class Retriever:
    """Base class for indexes dedicated to vector similarity retrieval"""

    def __init__(self, doc_path: str):
        self.doc_path = doc_path

    def create_db(self, chunk_size, chunk_overlap):
        raise NotImplementedError

    def find_neighbors(self, query: str, k: int):
        raise NotImplementedError


class FAISSRetriever(Retriever):
    """Class for FAISS index that build an index with OpenAI embeddings and finds k closest neighbors based on the query"""

    def __init__(self, doc_path: str, chunk_size=1000, chunk_overlap=100):
        super().__init__(doc_path)
        logger.debug("Starting index setup.")
        self.db = self.create_db(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.debug("Index setup was successful.")

    def create_db(self, chunk_size, chunk_overlap):
        """_summary_

        Args:
            chunk_size (int, optional): the length of the chunk
            chunk_overlap (int, optional): the number of characters overlap between chunks

        Returns:
            LangChain FAISS vetorstore
        """
        docs = self._read_docs()
        lc_docs = self._convert_to_lc_docs(docs)
        docs_split = self._split_docs(
            lc_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.debug("Creating FAISS index with OpenAI embeddings.")
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        db = FAISS.from_documents(docs_split, embedding_model)
        logger.debug("FAISS index with OpenAI embeddings was created.")
        return db

    def find_neighbors(self, query: str, k=4) -> list[Document]:
        """Based on a query finds k relevant documents in the index.

        Args:
            query (str): text request
            k (int, optional): Number of relevant documents we want to find. Defaults to 4

        Returns:
            list[Document]: The list of k relevant LangChain documents
        """
        return self.db.similarity_search(query, k=k)

    def _read_docs(self):
        logger.debug("Trying to read raw documents file.")
        with open(self.doc_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.debug("Raw documents file was found.")
        return docs

    def _convert_to_lc_docs(self, docs: list) -> list[Document]:
        logger.debug(
            "Converting documents to LangChain documents for convenient processing."
        )
        lc_docs = []
        for doc in docs:
            metadata = {
                "article_number": doc["article_number"],
                "article_summary": doc["article_summary"],
            }
            lc_doc = Document(page_content=doc["article_text"], metadata=metadata)
            lc_docs.append(lc_doc)
        logger.debug(
            "Documents were converted to LangChain documents format for convenient processing."
        )
        return lc_docs

    def _split_docs(
        self, lc_docs: list[Document], chunk_size=1000, chunk_overlap=100
    ) -> list[Document]:
        logger.debug("Splitting documents for vectorization.")
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs_split = text_splitter.split_documents(lc_docs)
        logger.debug("Documents were splitted.")
        return docs_split
