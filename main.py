import json
import uuid

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from app import main, setup_runnable, app, thread, stream_app_catch_tool_calls


@cl.on_chat_start
async def on_chat_start():
    app = setup_runnable()
    cl.user_session.set("runnable", app)

# Set up Chainlit app
@cl.on_message
async def on_message(message: cl.Message):
    # Create a human message
    human_message = HumanMessage(content=message.content)

    # Process the message through the existing workflow
    # await cl.Message(content="Processing your request...").send()
    response, tool_call_message = main(human_message)

    # Send the response back to the user
    if response != "":
        await cl.Message(content=response).send()

    while tool_call_message:
        verification_message = generate_verification_message(tool_call_message)
        await cl.Message(content=verification_message.content).send()

        user_response = await cl.AskUserMessage(content="Do you want to proceed with the tool? (y/n)").send()

        if user_response['content'].lower() == "exit":
            break

        snapshot = app.get_state(thread)
        snapshot.values["messages"] += [verification_message, HumanMessage(content=user_response['content'].lower())]

        if user_response['content'].lower() == "y":
            tool_call_message.id = str(uuid.uuid4())
            snapshot.values["messages"] += [tool_call_message]
            app.update_state(thread, snapshot.values, as_node="agent")
        else:
            app.update_state(thread, snapshot.values, as_node="__start__")

        tool_call_message = stream_app_catch_tool_calls(None, thread, app)


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


