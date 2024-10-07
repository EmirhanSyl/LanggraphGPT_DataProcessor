from typing import TypedDict

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from langgraph_agent.models.llama_model import ModelLLama
from langgraph_agent.tools.tools import ToolEditor, set_dataset
from langgraph_agent.tools.test.nonparametric import friedman_test, wilcoxon_signed_rank_test

import chainlit as cl

tool_editor = ToolEditor()
set_dataset(r"C:\Users\emirs\Documents\Projects\python\LanggraphGPT_DataProcessor\dataset\death_causes.csv")


llm = ModelLLama(tool_editor.get_test_tools()).llm  # Supervisor

hypothesis_agent_system_msg = SystemMessage(content="You are an agent that can test the hypothesis of the user."
                                                    " You have a lot of hypothesis tools. Use them if necessary.")
tool_node = ToolNode(tool_editor.get_test_tools())


class AgentState(TypedDict):
    messages: BaseMessage


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")


thread = {"configurable": {"thread_id": 1}}
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
inputs = {"messages": [hypothesis_agent_system_msg]}


def stream_app(new_message=None):
    response_message = None
    for chunk in app.stream(new_message, thread, stream_mode="values"):
        message = chunk["messages"][-1]
        message.pretty_print()
        response_message = message

    return response_message


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("runnable", app)
    stream_app(inputs)


@cl.on_message
async def on_message(message: cl.Message):
    human_message = HumanMessage(content=message.content)
    messages = {"messages": [human_message]}

    response = stream_app(messages)

    await cl.Message(content=response.content).send()

    # snapshot = app.get_state(thread)
    # print(snapshot.values)

