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

    # __________________________ NODES __________________________
    def generate_greeting(self, state):
        print("Greet")
        messages = state["messages"]

        hi_msg_prompt = (
            "You are a helpful AI agent for processing the data using tools. Your name is 'beeg'. Give a nice,"
            " friendly and helpful message to welcome the user. After that request to upload a csv dataset from user.")
        messages += [HumanMessage(content=hi_msg_prompt)]
        response = self.model.llm.invoke(messages)
        messages += [response]

        new_state = {
            "messages": messages,
            "last_message_type": MessageTypes.CHAT,
            "last_called_tool": [{}]
        }
        return new_state

    def generate_dataset_summary(self, state):
        messages = state["messages"]

        tool_call = {
            "name": "summarize_dataset",
            "args": {},
            "id": "tool_call_1"  # Unique identifier for the tool call
        }

        # Create the ToolInvocation object
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        response = self.tool_executor.invoke(action)
        print(response)
        response = json.dumps(response, indent=4)

        tool_message = ToolMessage(
            content=str(response),
            name=action.tool,
            tool_call_id=tool_call["id"]
        )

        # We return a list, because this will get added to the existing list
        messages += [tool_message]

        result_prompt = (f"Write a result message about user's dataset. The following dictionary is the summary of the "
                         f"dataset.\n{response}\nGive dataset summary to the user and examine the results accordingly")

        messages += [HumanMessage(content=result_prompt)]
        response = self.model.llm.invoke(messages)
        messages += [response]

        new_state = {
            "messages": messages,
            "last_message_type": MessageTypes.CHAT,
            "last_called_tool": [tool_call]
        }
        return new_state

    def report_missing_ratios(self, state):
        messages = state["messages"]

        tool_call = {
            "name": "calculate_missing_values",
            "args": {},
            "id": "tool_call_2"  # Unique identifier for the tool call
        }

        # Create the ToolInvocation object
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        response = self.tool_executor.invoke(action)
        response = json.dumps(response, indent=4)

        tool_message = ToolMessage(
            content=str(response),
            name=action.tool,
            tool_call_id=tool_call["id"]
        )

        # We return a list, because this will get added to the existing list
        messages += [tool_message]

        result_prompt = (f"Write a result message about user's dataset. The following dictionary is the missing value "
                         f"ratios of the dataset.\n{response}\nExamine the results for user. If missing value ratio of "
                         f"any column is bigger then 5%, warn the user about this data considered as manipulated")

        messages += [HumanMessage(content=result_prompt)]
        response = self.model.llm.invoke(messages)
        messages += [response]

        new_state = {
            "messages": messages,
            "last_message_type": MessageTypes.CHAT,
            "last_called_tool": state["last_called_tool"] + [tool_call]
        }
        return new_state

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



    # Define the function that determines whether to continue or not
    def should_continue(self, state):
        print("Conditional Edge")
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    # Define the function that calls the model
    def call_model(self, state):
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

    # Define the function to execute tools
    def call_tool(self, state):
        print("Action Node")
        tool_call_messages = state["last_called_tool"]

        # We construct a ToolInvocation from the function_call
        tool_call = tool_call_messages[-1].tool_calls[0]
        print(tool_call)
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)
        # We use the response to create a ToolMessage
        tool_message = ToolMessage(
            content=str(response), name=action.tool, tool_call_id=tool_call["id"]
        )

        # We return a list, because this will get added to the existing list
        return {"messages": [tool_message]}

    # Define the function that generates a verification message using the LLM
    def generate_dynamic_verification(self, state):
        print("Verification Node")
        messages = state["messages"]
        last_message = messages[-1]

        # Extract tool call information
        tool_call_info = json.dumps(last_message.tool_calls, indent=2)
        prompt = (f"Can you generate a verification message for informing the user about planning to use the "
                  f"following tool?\n\n{tool_call_info}")
        print(prompt)
        messages += [HumanMessage(content=prompt)]
        # Call the LLM to generate a dynamic verification message
        response = self.model.llm.invoke(messages)
        print(response)
        new_state = {
            "messages": [response],
            "last_message_type": MessageTypes.VERIFICATION,
        }
        return new_state

    def generate_tool_result_message(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            prompt = (f"Can you generate a verification message for informing the user about used tool results. "
                      f"Here is the result of the function call: \n\n{last_message}")
            messages += [HumanMessage(content=prompt)]
            response = self.model.llm.invoke(messages)
            print(response)
            new_state = {
                "messages": [response],
                "last_message_type": MessageTypes.CHAT,
            }
            return new_state

    # __________________________ WORKFLOW __________________________
    def setup_workflow(self):
        workflow = StateGraph(State)

        workflow.add_node("greet", self.generate_greeting)
        workflow.add_node("summary", self.generate_dataset_summary)
        workflow.add_node("report_missing", self.report_missing_ratios)
        workflow.add_node("handle_missings", self.handle_missings)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("generate_verification", self.generate_dynamic_verification)
        workflow.add_node("action", self.call_tool)
        workflow.add_node("action_result", self.generate_tool_result_message)

        workflow.add_edge(START, "greet")
        workflow.add_edge("greet", "summary")
        workflow.add_edge("summary", "report_missing")

        workflow.add_conditional_edges(
            "report_missing",
            self.should_handle_missings,
            {
                "handle": "handle_missings",
                "skip": "agent",
            },
        )

        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            # The keys are strings, and the values are other nodes.
            {
                "continue": "generate_verification",
                "end": END,
            },
        )

        workflow.add_edge("generate_verification", "action")
        workflow.add_edge("action", "action_result")
        workflow.add_edge("action_result", END)

        return workflow

