import json
import uuid
from typing import Optional, Tuple, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph_agent.workflow import setup_workflow
from langgraph.checkpoint.memory import MemorySaver

app = None
thread = {"configurable": {"thread_id": "3"}}


# Helper function to stream output from the graph
def stream_app_catch_tool_calls(inputs, thread) -> tuple[Optional[AIMessage], Optional[Any]]:
    """Stream app, catching tool calls."""
    global app
    tool_call_message = None
    response_message = None
    for event in app.stream(inputs, thread, stream_mode="values"):
        message = event["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_message = message
        else:
            message.pretty_print()
            response_message = message

    return tool_call_message, response_message


def setup_runnable():
    global app
    workflow = setup_workflow()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
    return app


def main(human_message= None):
    global thread
    inputs = [human_message]
    thread = {"configurable": {"thread_id": "3"}}
    tool_call_message, response_message = stream_app_catch_tool_calls({"messages": inputs}, thread)

    response = response_message.content if response_message else ""

    return response, tool_call_message

