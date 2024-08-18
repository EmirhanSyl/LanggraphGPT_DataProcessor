from langchain_ollama import ChatOllama

class ModelLLama:
    def __init__(self, tools: list = None):
        llm = ChatOllama(model="llama3.1:8b", temperature=0)
        if tools:
            llm = llm.bind_tools(tools)
        self.llm = llm
