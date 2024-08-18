import json
import uuid
from typing import Optional, Tuple, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph_agent.workflow import Workflow
from langgraph.checkpoint.memory import MemorySaver


class App:
    def __init__(self, thread_id: str = "0") -> None:
        self.thread = {"configurable": {"thread_id": thread_id}}
        self.wf = Workflow()
        self.workflow = self.wf.workflow_model
        self.memory = MemorySaver()
        self.app_runnable = self.workflow.compile(checkpointer=self.memory, interrupt_before=["action"])
        self.app_runnable.get_graph().draw_png("workflow_graph.png")

    # Helper function to stream output from the graph
    def stream_app_catch_tool_calls(self, inputs, thread) -> tuple[Optional[AIMessage], Optional[Any]]:
        """Stream app, catching tool calls."""
        tool_call_message = None
        response_message = None
        for event in self.app_runnable.stream(inputs, thread, stream_mode="values"):
            message = event["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_call_message = message
            else:
                message.pretty_print()
                response_message = message

        return tool_call_message, response_message

    def main(self, human_message=None):
        inputs = [human_message]
        tool_call_message, response_message = self.stream_app_catch_tool_calls({"messages": inputs}, self.thread)

        response = response_message.content if response_message else ""

        return response, tool_call_message

