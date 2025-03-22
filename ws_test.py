# ws_test.py
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    session_id = cl.user_session.get("id")
    print("Session started, id:", session_id)
    await cl.Message(content=f"Your session ID is: {session_id}").send()
    await cl.Message(content="Welcome! Send a message to start chatting.").send()

@cl.on_message
async def on_message(message: str):
    print("Received message:", message)
    await cl.Message(content=f"Echo: {message}").send()