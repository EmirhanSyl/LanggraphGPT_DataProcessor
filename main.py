import json
import uuid

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_agent.agent_state import MessageTypes
from app import App
from langgraph_agent.tools.tools import set_dataset

app = App()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("runnable", app.app_runnable)
    response = app.stream_app_catch_tool_calls({"messages": [HumanMessage(content="")]})
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content=response.content, accept=["text/csv"], max_files=1
        ).send()

    dataset = files[0]
    set_dataset(dataset.path)
    app.stream_app_catch_tool_calls()

    snapshot = app.app_runnable.get_state(app.thread)
    message = snapshot.values["messages"][-2]
    print(snapshot.values)
    await cl.Message(content=message.content).send()


# Set up Chainlit app
@cl.on_message
async def on_message(message: cl.Message):
    # Create a human message
    human_message = HumanMessage(content=message.content)
    response = app.stream_app_catch_tool_calls({"messages": [human_message]})

    snapshot = app.app_runnable.get_state(app.thread)
    print(snapshot.values)
    message_type = snapshot.values["last_message_type"]

    if message_type == MessageTypes.CHAT:
        await cl.Message(content=response.content).send()
    elif message_type == MessageTypes.VERIFICATION:
        actions = [
            cl.Action(name="approve_tool_use", value="approve", description="approve"),
            cl.Action(name="deny_tool_use", value="deny", description="deny"),
        ]
        await cl.Message(content=response.content, actions=actions).send()


# Handle button click event
@cl.action_callback("approve_tool_use")
async def on_action_approve(action):
    if action.value == "approve":
        snapshot = app.app_runnable.get_state(app.thread)
        tool_call_message = snapshot.values['messages'][-2]
        tool_call_message.id = str(uuid.uuid4())

        snapshot.values['messages'] += [HumanMessage(content="I approved to use this tool. Tool executed."), tool_call_message]
        app.app_runnable.update_state(app.thread, snapshot.values)
        await action.remove()

        response = app.stream_app_catch_tool_calls()
        await cl.Message(content=response.content).send()


@cl.action_callback("deny_tool_use")
async def on_action_deny(action):
    if action.value == "deny":
        snapshot = app.app_runnable.get_state(app.thread)
        snapshot.values['messages'] += [HumanMessage(content="I denied to use this tool. Tool didn't called.")]

        app.app_runnable.update_state(app.thread, snapshot.values, as_node="__start__")
        await action.remove()

        response = app.stream_app_catch_tool_calls({"messages": [HumanMessage(content="Generate a notification message about user denied the tool call.")]})
        await cl.Message(content=response.content).send()


async def send_chainlit_message(content):
    # Create and send the message
    message = cl.Message(content=content)
    await message.send()
