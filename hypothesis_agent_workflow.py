from typing import TypedDict


import chainlit as cl

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langgraph_agent.models.llama_model import ModelLLama, quantitative_prompt
from langgraph_agent.tools.tools import ToolEditor, set_dataset

tool_editor = ToolEditor()

llm = ModelLLama(tool_editor.get_correlation_tools() + tool_editor.get_regression_tools() + tool_editor.get_nonparametric_tools() + tool_editor.get_parametric_tools()).llm

hypothesis_agent_system_msg = SystemMessage(content=quantitative_prompt)
tool_node = ToolNode(tool_editor.get_correlation_tools() + tool_editor.get_regression_tools() + tool_editor.get_nonparametric_tools() + tool_editor.get_parametric_tools())


class AgentState(TypedDict):
    messages: BaseMessage


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        print(f"Tool call: {last_message.tool_calls}")
        return "tools"
    return END


def call_model(state: MessagesState):
    print(state)
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
thread_id = 0
memory = MemorySaver()

inputs = {"messages": [hypothesis_agent_system_msg]}


async def stream_app(new_message=None):
    response_message = None
    session_app = cl.user_session.get("runnable")
    session_thread = cl.user_session.get("thread")
    print(f"STARTED MF {id(session_app)}")
    # for chunk in session_app.stream(new_message, session_thread, stream_mode="values"):
    #     message = chunk["messages"][-1]
    #     message.pretty_print()
    #     response_message = message

    first = True
    gathered = ""
    async for msg, metadata in session_app.astream(new_message, config=session_thread, stream_mode="messages"):
        print(msg)
        print(metadata)
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="|", flush=True)

        if isinstance(msg, AIMessageChunk):
            if first:
                gathered = msg
                first = False
            else:
                gathered = gathered + msg

            if msg.tool_call_chunks:
                print(gathered.tool_calls)

    response_message = gathered
    return response_message


@cl.on_chat_start
async def on_chat_start():
    print("Session id:", cl.user_session.get("id"))
    global thread_id
    thread_id = thread_id + 1
    session_thread = {"configurable": {"thread_id": thread_id}}
    app = workflow.compile(checkpointer=memory)
    cl.user_session.set("runnable", app)
    cl.user_session.set("thread", session_thread)
    cl.user_session.set("inputs", inputs)

    async with cl.Step(name="Düşünce") as step:
        # Step is sent as soon as the context manager is entered
        step.output = "world"
        await cl.Message("test").send()


@cl.on_message
async def on_message(message: cl.Message):
    human_message = HumanMessage(content=message.content)
    inputs = cl.user_session.get("inputs")
    inputs["messages"].append(human_message)

    response = await stream_app(inputs)

    await cl.Message(content=response).send()

    # snapshot = app.get_state(thread)
    # print(snapshot.values)




