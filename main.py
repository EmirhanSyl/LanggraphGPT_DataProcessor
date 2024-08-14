import chainlit as cl
from langchain_core.messages import HumanMessage
from app import main


# Set up Chainlit app
@cl.on_message
async def on_message(message: cl.Message):
    # Create a human message
    human_message = HumanMessage(content=message.content)

    # Process the message through the existing workflow
    await cl.Message(content="Processing your request...").send()
    response = main(human_message)

    # Send the response back to the user
    await cl.Message(content=response).send()

