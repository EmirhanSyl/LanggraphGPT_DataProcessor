import json
import uuid

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_agent.agent_state import MessageTypes
from app import App

app = App()


@cl.on_chat_start
async def on_chat_start():
    hi_msg_prompt = ("You are a helpful AI agent for processing the data using tools. Your name is 'beeg'. Give a nice,"
                     " friendly and helpful message to welcome the user.")
    cl.user_session.set("runnable", app.app_runnable)
    response = app.stream_app_catch_tool_calls({"messages": [HumanMessage(content=hi_msg_prompt)]})

    initial_state = {
        "messages": [AIMessage(content=response.content)],
        "last_message_type": MessageTypes.CHAT,  # Initialize with a default value
        "last_called_tool": [{}]  # Initialize as an empty dictionary
    }

    # Start the workflow with the initial state
    app.app_runnable.update_state(app.thread, initial_state)
    await cl.Message(content=response.content).send()


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


# Helper function to construct message asking for verification
def generate_verification_message(message: AIMessage) -> AIMessage:
    """Generate "verification message" from message with tool calls."""
    serialized_tool_calls = json.dumps(
        message.tool_calls,
        indent=2,
    )
    return AIMessage(
        content=(
            "I plan to invoke the following tools, do you approve?\n\n"
            "Type 'y' if you do, anything else to stop.\n\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )


