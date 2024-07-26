import logging
import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from engine.retriever import FAISSRetriever

logger = logging.getLogger(__name__)

PATH_FOR_DOCS = os.getenv("PATH_FOR_DOCS")


class Chatbot:

    SYSTEM_TEMPLATE = """
You are a helpful assistant who answers questions about the GDPR Articles. Answer the user's questions based on the below context.
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

    def __init__(self):
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_TEMPLATE),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.chain = create_stuff_documents_chain(llm, prompt)
        self.memory = ChatMessageHistory()
        self.rag = FAISSRetriever(PATH_FOR_DOCS)

    def respond(self, human_msg: str) -> AIMessage:
        self.memory.add_user_message(human_msg)
        docs = self.rag.find_neighbors(human_msg)
        for d in docs:
            logger.debug("Found relevant document:\n %s", d)
        response = self.chain.invoke(
            {"messages": self.memory.messages, "context": docs}
        )
        self.memory.add_ai_message(response)
        return response
