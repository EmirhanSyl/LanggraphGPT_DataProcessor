import random

from langchain_ollama import ChatOllama

class ModelLLama:
    def __init__(self, tools: list = None):
        seed = random.randint(0, 2 ** 32 - 1)  # 32-bit random integer
        llm = ChatOllama(model="llama3-groq-tool-use", temperature=0, num_predict=-1, seed=seed)
        if tools:
            llm = llm.bind_tools(tools)
        self.llm = llm
