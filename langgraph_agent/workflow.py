import json

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolExecutor, ToolInvocation, ToolNode
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

from .agent_state import State, MessageTypes
from .tools.tools import ToolEditor, calculate_total_missing_values, generate_tool_calls_for_missing_values
from .models.llama_model import ModelLLama

import plotly.graph_objects as go

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
        return self.process_tool_task(state,message_prompt=greeting_prompt)

    def generate_dataset_summary(self, state):
        tool_call = {
            "name": "summarize_dataset",
            "args": {},
            "id": "tool_call_1"
        }
        dataset_summary_prompt = (
            "Write a result message about the user's dataset. The following dictionary is the summary of the dataset.\n"
            "Give a dataset summary to the user and examine the results accordingly.\n"
        )
        return self.process_tool_task(state,message_prompt=dataset_summary_prompt, tool_call=tool_call)

    def report_missing_ratios(self, state):
        tool_call = {
            "name": "calculate_missing_values",
            "args": {},
            "id": "tool_call_2"
        }
        missing_ratio_prompt = (
            "Write a result message about the user's dataset. The following dictionary contains the missing value ratios.\n"
            "Examine the results for the user. If the missing value ratio of any column is greater than 5%, "
            "warn the user that the data might be manipulated."
        )
        return self.process_tool_task(state, tool_call=tool_call, message_prompt=missing_ratio_prompt)

    def should_handle_missings(self, state):
        if calculate_total_missing_values() > 0:
            return "handle"
        else:
            return "skip"

    def handle_missings(self, state):
        tools = generate_tool_calls_for_missing_values()
        tool_call_messages = []
        messages = state["messages"]

        for tool_prompt in tools:
            content = f"To handle missing values, use the following tool call:\n{tool_prompt}"
            messages += [HumanMessage(content=content)]
            messages += [self.model.llm.invoke(messages)]
            tool_call_messages += messages[-1]

        new_state = {
            "messages": messages,
            "last_message_type": MessageTypes.TOOL_USE,
        }
        print(f"State:{state}")
        return new_state


    # __________________________ WORKFLOW __________________________
    def setup_workflow(self):
        workflow = StateGraph(State)

        workflow.add_node("greet", self.generate_greeting)
        workflow.add_node("summary", self.generate_dataset_summary)
        workflow.add_node("report_missing", self.report_missing_ratios)
        workflow.add_node("handle_missings", self.handle_missings)

        workflow.add_edge(START, "greet")
        workflow.add_edge("greet", "summary")
        workflow.add_edge("summary", "report_missing")

        workflow.add_conditional_edges(
            "report_missing",
            self.should_handle_missings,
            {
                "handle": "handle_missings",
                "skip": END,
            },
        )

        workflow.add_edge("handle_missings", END)

        return workflow

