import json
import logging
import os
import re

from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PATH_TO_SUMMARIES = os.getenv("PATH_TO_SUMMARIES")
PATH_FOR_DOCS = os.getenv("PATH_FOR_DOCS")


class PDFRetriever:
    """Base class for PDF files parsing."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.text = self._retrieve_text(filepath)

    def _retrieve_text(self, filepath) -> str:
        logger.info("Starting text retrieval from the PDF file.")
        text = extract_text(filepath)
        logger.info("Text was successfully retrieved from the PDF file!")
        return text

    def parse_pdf(self) -> None:
        raise NotImplementedError


class GDPRRetriever(PDFRetriever):
    """PDF parser dedicated for parsing GDPR PDF file."""

    def parse_pdf(self) -> None:
        docs = self._process_text_to_docs()
        self._save_docs_to_json(docs)

    def _process_text_to_docs(self) -> list[dict]:
        """Splits a raw text to separate articles, runs preprocessing, combines it with summary and enumerates articles.
        In the end saves it in the json format.

        Returns:
            dict: list of articles with a number of article, it's summary and text
        """
        logger.info("Starting preparing documents from the text.")
        logger.info("Separating articles.")
        articles = self.text.split("EN\n\nArticle ")[1:]
        logger.info("Articles are separated. Found %s articles", len(articles))
        logger.info("Starting removing redundant components from articles.")
        articles = [self._preprocess_article(x) for x in articles]
        logger.info("Redundant components from articles were removed.")
        docs = self._prepare_docs(articles)
        logger.info("Documents are ready and saved to %s", PATH_FOR_DOCS)
        return docs

    def _preprocess_article(self, article: str):
        """Prepocessing of particular articles from GDPR PDF file. Removes the first page of the article
        that is redundant, removes footers and page numbers.

        Args:
            article (str): initial version of the article

        Returns:
            str: the article after preprocessing
        """
        article = article.split("\n\nArticle ", maxsplit=1)[1]
        article = article.split(".\n\n", maxsplit=1)[1]
        article = re.sub(
            r"\s*GDPR\s+training,\s+consulting\s+and\s+DPO\s+outsourcing\s*page\s+\d+\s*/\s*\d+\s*\x0cwww\.gdpr-text\.com/en\s*",
            "",
            article,
            flags=re.DOTALL,
        )
        article = re.sub(
            r"\s*GDPR\s+training,\s+consulting\s+and\s+DPO\s+outsourcing\s*page\s+\d+\s*/\s*\d+\s+Powered\s+by\s+TCPDF\s*\(www\.tcpdf\.org\)\s*\x0c\s*",
            "",
            article,
            flags=re.DOTALL,
        )
        return article

    def _save_docs_to_json(self, docs: dict):
        with open(PATH_FOR_DOCS, "w", encoding="utf-8") as outfile:
            json.dump(docs, outfile)

    def _prepare_docs(self, articles: list[str]) -> list[dict]:
        with open(PATH_TO_SUMMARIES, encoding="utf-8") as f:
            summaries = f.readlines()
        docs = [
            {
                "article_number": i + 1,
                "article_summary": summaries[i],
                "article_text": articles[i],
            }
            for i in range(len(summaries))
        ]
        return docs


if __name__ == "__main__":
    PATH_TO_PDF = os.getenv("PATH_TO_PDF")
    retriever = GDPRRetriever(filepath=PATH_TO_PDF)
    retriever.parse_pdf()
