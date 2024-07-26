import json
import os

from engine.chatbot import Chatbot
from engine.retriever import FAISSRetriever
from utils.text_retrieval import GDPRRetriever

PATH_TO_PDF = os.getenv("PATH_TO_PDF")
PATH_FOR_DOCS = os.getenv("PATH_FOR_DOCS")


def test_pdf_parser():
    """Checks whether parser correctly parsed 21 article from GDPR PDF."""

    retriever = GDPRRetriever(filepath=PATH_TO_PDF)
    retriever.parse_pdf()

    with open(PATH_FOR_DOCS, "r", encoding="utf-8") as f:
        docs = json.load(f)
    assert len(docs) == 21


def test_retriever_db():
    """Checks Retriever db can be created."""
    rag = FAISSRetriever(doc_path=PATH_FOR_DOCS)
    assert rag.db


def test_retriever_search():
    """Checks Retriever is able to find relevant docs."""
    query = "What is the principle of transparency stands for?"
    rag = FAISSRetriever(doc_path=PATH_FOR_DOCS)
    n = rag.find_neighbors(query, k=4)
    assert len(n) == 4
    n = rag.find_neighbors(query, k=2)
    assert len(n) == 2


def test_bot_response():
    """Checks whether bot is able to respond a question."""
    bot = Chatbot()
    question = "What is the principle of transparency stands for?"
    response = bot.respond(question)
    assert response
    assert type(response) == str


def test_bot_response_irrelevant():
    """Checks whether bot answers I don't know when the question is irrelevant to GDPR."""
    bot = Chatbot()
    question = "How big is the moon?"
    response = bot.respond(question)
    assert response == "I don't know."
