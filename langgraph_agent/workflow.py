import json

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolExecutor, ToolInvocation, ToolNode
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

from .agent_state import State, MessageTypes
from .tools.tools import ToolEditor
from .models.llama_model import ModelLLama


class Workflow:
    def __init__(self):
        self.tool_editor = ToolEditor()
        self.tool_executor = self.tool_editor.tool_executor
        self.model = ModelLLama(self.tool_editor.tools)
        self.workflow_model = self.setup_workflow()

    # __________________________ NODES __________________________
    # Define the function that determines whether to continue or not
    def should_continue(self, state):
        print("Conditional Edge")
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            state['last_message_type'] = MessageTypes.TOOL_USE
            state['last_called_tool'] = last_message.tool_calls
            return "end"
        else:
            state['last_message_type'] = MessageTypes.CHAT
            return "continue"

    # Define the function that calls the model
    def call_model(self, state):
        print("Agent Node")
        messages = state["messages"]
        response = self.model.llm.invoke(messages)
        new_state = {
            "messages": [response],
            "last_message_type": MessageTypes.CHAT,
        }
        return new_state

    # Define the function to execute tools
    def call_tool(self, state):
        print("Action Node")
        messages = state["messages"]
        last_message = messages[-1]

        # We construct a ToolInvocation from the function_call
        tool_call = last_message.tool_calls[0]
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
        prompt = (f"I am planning to use the following tool. Can you generate a verification"
                  f" message for the user?\n\n{tool_call_info}")

        # Call the LLM to generate a dynamic verification message
        response = self.model.llm.invoke([HumanMessage(content=prompt)])
        new_state = {
            "messages": [response],
            "last_message_type": MessageTypes.VERIFICATION,
        }
        return new_state

    # __________________________ WORKFLOW __________________________
    def setup_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("generate_verification", self.generate_dynamic_verification)
        workflow.add_node("action", self.call_tool)

        workflow.add_edge(START, "agent")
        workflow.add_edge("generate_verification", "action")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            # The keys are strings, and the values are other nodes.
            {
                "continue": "generate_verification",
                "end": END,
            },
        )
        return workflow

