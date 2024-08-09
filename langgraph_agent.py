import json
import operator
from typing import TypedDict, Union, Annotated

from IPython.display import Image
from langchain.agents import create_tool_calling_agent, create_openai_tools_agent
from langchain_ollama import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

db = [{
    'username': 'ahmet',
    'password': '123',
}]


class AgentState(TypedDict):
    input: str
    agent_out: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


@tool("validate")
def validate_user_tool(username: str, password: str) -> bool:
    """Validate user using username and password.

    Args:
        username: (int) the username of the user.
        password: (str) user's password.
    """
    for entry in db:
        if entry['username'] == username and entry['password'] == password:
            return True
    return False


@tool("final_answer")
def final_answer_tool(answer: str, logged_in: bool):
    """Returns a natural language response to the user in 'answer',
    and a 'logged_in' which represents if the user logged in successfully."""
    return ""


# Initilize the agent
llm = ChatOllama(
    model="llama3-groq-tool-use",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a password validators for the users. Validate Passwords and if user is valid, say 'holla!'",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

query_agent_runnable = create_openai_tools_agent(
    llm=llm,
    tools=[final_answer_tool, validate_user_tool],
    prompt=prompt
)

inputs = {
    "input": "Do you verify that a user named 'ahmet' with the password '123' exists?",
    "intermediate_steps": []
}
agent_out = query_agent_runnable.invoke(inputs)
print(agent_out)

# Define Nodes For Graph
def run_query_agent(state: list):
    print("> run_query_agent")
    agent_out = query_agent_runnable.invoke(state)
    return {"agent_out": agent_out}


def execute_validation(state: list):
    print("> execute_validation")
    action = state["agent_out"]
    tool_call = action[-1].message_log[-1].additional_kwargs["tool_calls"][-1]
    out = validate_user_tool.invoke(
        json.loads(tool_call["function"]["arguments"])
    )
    return {"intermediate_steps": [{"validate": str(out)}]}


def router(state: list):
    print("> router")
    if isinstance(state["agent_out"], list):
        return state["agent_out"][-1].tool
    else:
        return "error"


final_answer_llm = llm.bind_tools([final_answer_tool], tool_choice="final_answer")


def rag_final_answer(state: list):
    print("> final_answer")
    query = state["input"]
    context = state["intermediate_steps"][-1]

    prompt = f"""You are a helpful assistant, answer the user's question using the
    context provided.

    CONTEXT: {context}

    QUESTION: {query}
    """
    out = final_answer_llm.invoke(prompt)
    function_call = out.additional_kwargs["tool_calls"][-1]["function"]["arguments"]
    return {"agent_out": function_call}


# we use the same forced final_answer LLM call to handle incorrectly formatted
# output from our query_agent
def handle_error(state: list):
    print("> handle_error")
    query = state["input"]
    prompt = f"""You are a helpful assistant, answer the user's question.

    QUESTION: {query}
    """
    out = final_answer_llm.invoke(prompt)
    function_call = out.additional_kwargs["tool_calls"][-1]["function"]["arguments"]
    return {"agent_out": function_call}


graph = StateGraph(AgentState)

# we have four nodes that will consume our agent state and modify
# our agent state based on some internal process
graph.add_node("query_agent", run_query_agent)
graph.add_node("validate", execute_validation)
graph.add_node("error", handle_error)
graph.add_node("rag_final_answer", rag_final_answer)

# our graph will always begin with the query agent
graph.set_entry_point("query_agent")

# conditional edges are controlled by our router
graph.add_conditional_edges(
    "query_agent",  # where in graph to start
    router,  # function to determine which node is called
)
graph.add_edge("validate", "rag_final_answer")
graph.add_edge("error", END)
graph.add_edge("rag_final_answer", END)

runnable = graph.compile()
Image(runnable.get_graph().draw_png())

