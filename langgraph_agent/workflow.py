import json

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolExecutor, ToolInvocation, ToolNode
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

from .agent_state import State, MessageTypes
from .tools.tools import (ToolEditor, check_preprocess_needed, get_dataset_sample, calculate_total_missing_values,
                          should_handle_outliers)
from .models.llama_model import ModelLLama


class Workflow:
    def __init__(self):
        self.tool_editor = ToolEditor()
        self.tool_executor = self.tool_editor.tool_executor
        self.model = ModelLLama(self.tool_editor.tools)
        self.workflow_model = self.setup_workflow()

    def process_tool_task(self, state, message_prompt: str, tool_call: dict = None) -> dict:
        messages = state["messages"]

        # Create and invoke the tool
        if tool_call:
            action = ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"]
            )
            response = self.tool_executor.invoke(action)
            response_str = json.dumps(response, indent=4)

            tool_message = ToolMessage(
                content=response_str,
                name=action.tool,
                tool_call_id=tool_call["id"]
            )
            message_prompt = message_prompt + response_str
            messages += [tool_message]

        messages += [HumanMessage(content=message_prompt)]

        # Invoke the LLM
        response = self.model.llm.invoke(messages)
        messages += [response]

        # Update the state with the new messages and tool call
        new_state = {
            "messages": messages,
            "last_message_type": MessageTypes.CHAT,
            "last_called_tool": [tool_call]
        }

        return new_state

    # __________________________ NODES __________________________
    def generate_greeting(self, state):
        greeting_prompt = (
            "You are a helpful AI agent for processing the data using tools. Your name is 'Beeg'. "
            "Give a nice, friendly, and helpful message to welcome the user. "
            "After that, request to upload a CSV dataset from the user."
        )
        # No tool is used for greeting, so tool_call is None
        return self.process_tool_task(state, message_prompt=greeting_prompt)

    def generate_dataset_summary(self, state):
        tool_call = {
            "name": "summarize_dataset",
            "args": {},
            "id": "tool_call_1"
        }
        dataset_sample = get_dataset_sample()
        dataset_summary_prompt = (
            f"Write a result message about the user's dataset. Here is the part of the dataset: \n'{dataset_sample}'\n "
            f"The following dictionary is the summary of the dataset. If the missing value ratio of any column is "
            f"greater than 5%, warn the user that the data might be manipulated.\n"
            f"Give a very detailed dataset summary to the user and examine the results accordingly.\n"
        )
        return self.process_tool_task(state,message_prompt=dataset_summary_prompt, tool_call=tool_call)

    def check_preprocess_needed(self, state):
        return check_preprocess_needed()

    def start_preprocess(self, state):
        process_details = """This function follows a detailed multistep process to manage missing values and outliers for different types
    of columns (numeric, string, and datetime). It modifies the global 'dataset' DataFrame in place by applying
    the following operations:

    Steps:
    1. **Type Conversion**:
       - For all columns, if the column is of 'Object' dtype, it is checked whether it can be converted
         to either 'String' or 'Datetime'. Conversion attempts:
         - Convert to 'datetime' using `pd.to_datetime()`. If the conversion fails (ValueError or TypeError),
           the column is converted to a 'String' type instead.

    2. **Missing Value Ratio Check**:
       - For each column in the DataFrame, the percentage of missing values is calculated.
       - If a column has more than 35% missing values, it is dropped from the DataFrame completely.

    3. **Handling Numeric Columns**:
       - For each numeric column in the dataset:
         - A normality test (Shapiro-Wilk) is conducted to determine if the column is normally distributed.
         - If the column is normally distributed, outliers are detected using the Z-score method (values with Z > 3
           are considered outliers).
         - If the column is not normally distributed, the IQR (Interquartile Range) method is used for outlier detection.
         - The ratio of outliers to total values is calculated:
           - If the outlier ratio is greater than 10%, missing values in the column are filled with the **median**.
           - Otherwise, missing values are filled with the **mean** of the column.

    4. **Handling String Columns**:
       - For each string column (dtype 'Object' or 'String'), missing values are filled with the constant value
         "unknown".

    5. **Handling Datetime Columns**:
       - For each datetime column:
         - Datetime values are converted to numeric timestamps for outlier detection.
         - Based on the normality of the column, outliers are detected using either the Z-score or IQR method.
         - The ratio of outliers to total values is calculated:
           - If the outlier ratio is greater than 15%, missing values in the column are filled with the **median**
             of the column.
           - If the outlier ratio is less than 15%, the column is checked to see if it represents a time series:
             - A time series is determined by checking if the data is sorted by time and has a regular frequency.
             - If the column is a time series, missing values are filled using **forward fill** (ffill) or
               **backward fill** (bfill).
             - If it is not a time series, missing values are filled with the **mean** of the column."""
        greeting_prompt = (
            f"Write an information message about how preprocessing steps will be done. Here I'm giving you to detailed "
            f"explanation:\n{process_details}"
        )
        return self.process_tool_task(state, message_prompt=greeting_prompt)

    def should_handle_missings(self, state):
        return "handle" if calculate_total_missing_values() > 0 else "skip"

    def handle_missing(self, state):
        tool_call = {
            "name": "handle_missing_values",
            "args": {},
            "id": "tool_call_3"
        }
        missing_ratio_prompt = (
            "According to the following tool call result message, write an information message to the user about what "
            "happened after missing values filled and how was the process."
        )
        return self.process_tool_task(state, tool_call=tool_call, message_prompt=missing_ratio_prompt)

    def should_handle_outliers(self, state):
        return should_handle_outliers()

    def handle_outliers(self, state):
        tool_call = {
            "name": "handle_outliers",
            "args": {},
            "id": "tool_call_4"
        }
        missing_ratio_prompt = (
            "According to the following tool call result message, write an information message to the user about what "
            "happened after outliers handled and how was the process."
        )
        return self.process_tool_task(state, tool_call=tool_call, message_prompt=missing_ratio_prompt)

    def ask_to_model(self, state):
        print("Agent Node")
        messages = state["messages"]
        tool_calls = state["last_called_tool"]
        response = self.model.llm.invoke(messages)
        last_msg_type = MessageTypes.CHAT
        if response.tool_calls:
            last_msg_type = MessageTypes.TOOL_USE
            tool_calls += [response]

        new_state = {
            "messages": [response],
            "last_message_type": last_msg_type,
            "last_called_tool": tool_calls,
        }
        return new_state

    # __________________________ WORKFLOW __________________________
    def setup_workflow(self):
        workflow = StateGraph(State)

        workflow.add_node("greet", self.generate_greeting)
        workflow.add_node("dataset_summary", self.generate_dataset_summary)
        workflow.add_node("start_preprocess", self.start_preprocess)
        workflow.add_node("handle_missing", self.handle_missing)
        workflow.add_node("handle_outliers", self.handle_outliers)
        workflow.add_node("ask_to_model", self.ask_to_model)

        workflow.add_edge(START, "greet")
        workflow.add_edge("greet", "dataset_summary")
        workflow.add_edge("dataset_summary", "start_preprocess")

        workflow.add_conditional_edges(
            "dataset_summary",
            self.check_preprocess_needed,
            {
                "preprocess": "start_preprocess",
                "skip": "ask_to_model",
            },
        )

        workflow.add_conditional_edges(
            "start_preprocess",
            self.should_handle_missings,
            {
                "handle": "handle_missing",
                "skip": "handle_outliers",
            },
        )

        workflow.add_conditional_edges(
            "handle_missing",
            self.should_handle_outliers,
            {
                "handle": "handle_outliers",
                "skip": "ask_to_model",
            },
        )

        workflow.add_edge("handle_outliers", "ask_to_model")
        workflow.add_edge("ask_to_model", "ask_to_model")

        return workflow

