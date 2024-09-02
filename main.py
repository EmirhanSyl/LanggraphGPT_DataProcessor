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

    await cl.Message(content="Preparing dataset summary...").send()
    dataset = files[0]
    set_dataset(dataset.path)
    app.stream_app_catch_tool_calls()

    snapshot = app.app_runnable.get_state(app.thread)
    print(snapshot.values)
    tool_message = snapshot.values["messages"][-3]
    tool_message_json = json.loads(tool_message.content)
    await cl.Message(content=tool_message_json).send()

    result_message = snapshot.values["messages"][-1]
    await cl.Message(content=result_message.content).send()

    actions = [
        cl.Action(name="approve_preprocess", value="approve", description="approve"),
        cl.Action(name="deny_preprocess", value="deny", description="deny"),
    ]
    await cl.Message(content="Do you want to continue with preprocessing steps?", actions=actions).send()


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


@cl.action_callback("approve_preprocess")
async def on_action_preprocessing_approve(action):
    app.stream_app_catch_tool_calls()
    await action.remove()

    snapshot = app.app_runnable.get_state(app.thread)

    tool_message = snapshot.values["messages"][-3]
    tool_message_json = json.loads(tool_message.content)
    await cl.Message(content=tool_message_json).send()

    result_message = snapshot.values["messages"][-1]
    await cl.Message(content=result_message.content).send()
    app.stream_app_catch_tool_calls()

@cl.action_callback("deny_preprocess")
async def on_action_preprocessing_deny(action):
    snapshot = app.app_runnable.get_state(app.thread)
    app.app_runnable.update_state(app.thread, snapshot.values, as_node="__end__")

    app.stream_app_catch_tool_calls()
    await action.remove()


async def send_chainlit_message(content):
    # Create and send the message
    message = cl.Message(content=content)
    await message.send()
