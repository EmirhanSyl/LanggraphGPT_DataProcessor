from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

import pandas as pd
df = pd.read_csv("dataset/death_causes.csv")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_dataset_summary(query: str):
    """
    Generate a summary of the dataset.

    This function retrieves a dataset,
    computes descriptive statistics for all numerical columns in the dataset,
    and returns the summary as a formatted string.

    Parameters:
    ----------
    query : str
        The query should be a valid string that can be used to fetch the relevant subset of the dataset.

    Returns:
    -------
    str
        A string representation of the summary statistics for the dataset,
        including metrics such as count, mean, standard deviation, min, max,
        and quartiles for each numerical column.
    """
    return df.describe().to_string()


@tool
def handle_missing_values(column_name: str) -> str:
    """
    Replace missing values in the specified column with the column's mean value.

    This function calculates the mean of the specified column, replaces any missing
    values (NaNs) with this mean, and returns a summary string indicating the mean
    value and the number of missing values that were replaced.

    Parameters:
    ----------
    column_name : str
        The name of the column in which missing values will be replaced with the mean value.

    Returns:
    -------
    str
        A string summarizing the operation, including the mean value used for replacement
        and the number of missing values that were filled.
    """
    mean_value = df[column_name].mean()
    missing_count = df[column_name].isna().sum()
    df[column_name].fillna(mean_value, inplace=True)

    return (f"The mean value for '{column_name}' is {mean_value:.2f}. "
            f"Replaced {missing_count} missing values with this mean.")


tools = [get_dataset_summary, handle_missing_values]
tool_executor = ToolExecutor(tools)

# Set up the model
llm = ChatOllama(
    model="llama3-groq-tool-use",
    temperature=0,
)
llm = llm.bind_tools(tools)


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

workflow.add_edge("action", "agent")
memory = MemorySaver()

app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
thread = {"configurable": {"thread_id": "2"}}
inputs = [HumanMessage(content="can you fill the missing values on the 'CAUSE' column in the dataset?")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()