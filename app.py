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
        self.app_runnable = self.workflow.compile(checkpointer=self.memory,
                                                  interrupt_before=["dataset_summary", "start_preprocess", "action",
                                                                    "ask_to_model"],
                                                  interrupt_after=["start_preprocess", "handle_missing",
                                                                   "handle_outliers"])
        self.app_runnable.get_graph().draw_png("workflow_graph.png")

    # Helper function to stream output from the graph
    def stream_app_catch_tool_calls(self, inputs=None) -> Optional[AIMessage]:
        """Stream app, catching tool calls."""
        model_input = {"messages": [inputs]}
        response_message = None
        for event in self.app_runnable.stream(inputs, self.thread, stream_mode="values"):
            message = event["messages"][-1]
            message.pretty_print()
            response_message = message

        return response_message

