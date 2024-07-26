import json
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


class Retriever:
    def __init__(self, doc_path: str):
        self.doc_path = doc_path

    def create_db(self):
        raise NotImplementedError

    def find_neighbours(self, k: int):
        raise NotImplementedError


class FAISSRetriever(Retriever):
    def __init__(self, doc_path: str, chunk_size=1000, chunk_overlap=100):
        super().__init__(doc_path)
        self.db = self.create_db(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def create_db(self, chunk_size=1000, chunk_overlap=100):
        docs = self._read_docs()
        lc_docs = self._convert_to_lc_docs(docs)
        docs_split = self._split_docs(
            lc_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        db = FAISS.from_documents(docs_split, embedding_model)
        return db

    def find_neighbours(self, query: str, k=4):
        return self.db.similarity_search(query, k=k)

    def _read_docs(self):
        with open(self.doc_path, "r") as f:
            docs = json.load(f)

        return docs

    def _convert_to_lc_docs(self, docs: list) -> list[Document]:
        lc_docs = []
        for doc in docs:
            metadata = {
                "article_number": doc["article_number"],
                "article_summary": doc["article_summary"],
            }
            lc_doc = Document(page_content=doc["article_text"], metadata=metadata)
            lc_docs.append(lc_doc)
        return lc_docs

    def _split_docs(
        self, lc_docs: list[Document], chunk_size=1000, chunk_overlap=100
    ) -> list[Document]:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs_split = text_splitter.split_documents(lc_docs)
        return docs_split
