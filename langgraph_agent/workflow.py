from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolExecutor, ToolInvocation, ToolNode
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

from .agent_state import State
from .tools.tools import get_tools
from .models.llama_model import setup_llm

tools = get_tools()
tool_executor = ToolExecutor(tools)
llm = setup_llm()


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct a ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}


def setup_workflow():
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)

    workflow.add_edge(START, "agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        {
            "continue": "action",
            "end": END,
        },
    )
    return workflow
