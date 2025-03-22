# test.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from chainlit.utils import mount_chainlit
from chainlit.context import init_ws_context
from chainlit.session import WebsocketSession
import chainlit as cl

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World from the main FastAPI app"}

@app.get("/hello/{session_id}")
async def hello(request: Request, session_id: str):
    ws_session = WebsocketSession.get_by_id(session_id=session_id)
    if not ws_session:
        raise HTTPException(status_code=404, detail="Websocket session not found.")
    init_ws_context(ws_session)
    await cl.Message(content="Hello from custom endpoint!").send()
    return {"message": "Data sent to websocket client"}

mount_chainlit(app=app, target="ws_test.py", path="/chainlit")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=5555, reload=True)