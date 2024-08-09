import operator
from typing import TypedDict, Union, Annotated

from langchain.agents import create_tool_calling_agent, create_openai_tools_agent
from langchain_ollama import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

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