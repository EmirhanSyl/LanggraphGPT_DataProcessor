from langchain_ollama import ChatOllama


def setup_llm(tools: list = None):
    llm = ChatOllama(model="llama3-groq-tool-use", temperature=0)
    if tools:
        llm = llm.bind_tools(tools)
    return llm

