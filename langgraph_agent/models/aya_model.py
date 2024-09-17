from langchain_ollama import ChatOllama

class ModelAya:
    def __init__(self):
        llm = ChatOllama(model="aya", temperature=0, num_predict=-1)
        self.llm = llm
