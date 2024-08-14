import json
import uuid
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph_agent.workflow import setup_workflow
from langgraph.checkpoint.memory import MemorySaver


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


# Helper function to stream output from the graph
def stream_app_catch_tool_calls(inputs, thread, app) -> Optional[AIMessage]:
    """Stream app, catching tool calls."""
    tool_call_message = None
    for event in app.stream(inputs, thread, stream_mode="values"):
        message = event["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_message = message
        else:
            message.pretty_print()

    return tool_call_message


def main():
    workflow = setup_workflow()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

    inputs = [HumanMessage(content="can you fill the missing values on the column named 2019 in the dataset?")]
    thread = {"configurable": {"thread_id": "3"}}
    tool_call_message = stream_app_catch_tool_calls({"messages": inputs}, thread, app)

    while tool_call_message:
        verification_message = generate_verification_message(tool_call_message)
        verification_message.pretty_print()
        input_message = HumanMessage(input())
        if input_message.content == "exit":
            break
        input_message.pretty_print()

        snapshot = app.get_state(thread)
        snapshot.values["messages"] += [verification_message, input_message]

        if input_message.content == "y":
            tool_call_message.id = str(uuid.uuid4())
            snapshot.values["messages"] += [tool_call_message]
            app.update_state(thread, snapshot.values, as_node="agent")
        else:
            app.update_state(thread, snapshot.values, as_node="__start__")

        tool_call_message = stream_app_catch_tool_calls(None, thread)


if __name__ == "__main__":
    main()
