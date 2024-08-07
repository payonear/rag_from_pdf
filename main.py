import logging
import time

from engine.chatbot import Chatbot

logging.getLogger("langchain_text_splitters").setLevel(level=logging.ERROR)

if __name__ == "__main__":
    print("[AI Assistant]: Hi, wait couple of seconds till I setup all the components!")
    time.sleep(3)
    bot = Chatbot()
    print("\n")
    print("[AI Assistant]: It's all set! I'm ready to help! Ask your GDRP question.")
    print("\n")
    time.sleep(2)
    while True:
        question = input("[Your question]: ")
        output = bot.respond(question)
        print("\n")
        print(f"[AI Assistant]: {output}")
        print("\n")
