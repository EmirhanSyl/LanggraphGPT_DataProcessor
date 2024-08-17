from langchain_ollama import ChatOllama

class ModelLLama:
    def __init__(self, tools: list = None):
        llm = ChatOllama(model="llama3-groq-tool-use", temperature=0)
        if tools:
            llm = llm.bind_tools(tools)
        self.llm = llm

def setup_llm(tools: list = None):
    llm = ChatOllama(model="llama3-groq-tool-use", temperature=0)
    if tools:
        llm = llm.bind_tools(tools)
    return llm

