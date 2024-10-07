import functools
import operator

from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import Unpack

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel
from typing import Literal, TypedDict, Annotated, Sequence

from langgraph.prebuilt import create_react_agent
from langgraph_agent.models.llama_model import ModelLLama
from langgraph_agent.tools.tools import ToolEditor, set_dataset

members = ["Hypothesis_Tester", "Dataset_Summarizer"]
options = ["FINISH"] + members
tool_editor = ToolEditor()
set_dataset(r"C:\Users\emirs\Documents\Projects\python\LanggraphGPT_DataProcessor\dataset\death_causes.csv")

llm = ModelLLama().llm  # Supervisor

hypothesis_agent_system_msg = SystemMessage(content="You are an agent that can test the hypothesis of the user."
                                                    " You have a lot of hypothesis tools. Use them if necessary. If "
                                                    "you need more information about users dataset, other agents will "
                                                    "provide you to necessary information, ")
hypothesis_agent = create_react_agent(llm, tools=tool_editor.hypothesis_tools,
                                      state_modifier=hypothesis_agent_system_msg)  # Hypothesis agent

dataset_summarizer_system_msg = SystemMessage(content="You are an agent that can provide information about user's "
                                                      "dataset. You have some summarizer tools. Use them if necessary. "
                                                      "If you dont have necessary information about users dataset, "
                                                      "provide obtained information,")
dataset_summarizer_agent = create_react_agent(llm, tools=tool_editor.dataset_summarizer_tools,
                                              state_modifier=dataset_summarizer_system_msg)  # Dataset Summarizer


class RouteResponse(BaseModel):
    next: Literal['START', 'FINISH', 'Hypothesis_Tester', 'Dataset_Summarizer']


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}


def empty_node(state):
    ...


def supervisor_agent():
    system_prompt = (
        f"You are a supervisor tasked with managing a conversation between the"
        f" following workers:  {members}. Given the following user request,"
        f" respond with the worker to act next. Each worker will perform a"
        f" task and respond with their results and status. When finished,"
        f" respond with FINISH."
    )
    # messages = state["messages"]
    system_question = f"Given the conversation above, who should act next?" \
                      f" Or should we FINISH? Select one of: {options}"

    # prompt ={"messages": [SystemMessage(content=system_prompt)] + messages + [SystemMessage(content=system_question)]}
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),  # Placeholder for previous conversation
            ("system", system_question),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
            prompt
            | llm.with_structured_output(RouteResponse)
    )
    return supervisor_chain


hypothesis_node = functools.partial(agent_node, agent=hypothesis_agent, name="Hypothesis Tester")
dataset_summarizer_node = functools.partial(agent_node, agent=dataset_summarizer_agent, name="Dataset Summarizer")

workflow = StateGraph(AgentState)
workflow.add_node("Hypothesis_Tester", hypothesis_node)
workflow.add_node("Dataset_Summarizer", dataset_summarizer_node)
workflow.add_node("supervisor", supervisor_agent())
workflow.add_node("START", empty_node)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
conditional_map["START"] = "START"
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# workflow.add_conditional_edges(
#             "supervisor",
#             select_agent,
#             {
#                 "Hypothesis_Tester": "Hypothesis_Tester",
#                 "Dataset_Summarizer": "Dataset_Summarizer",
#                 "FINISH": END,
#             },
#         )
workflow.add_edge(START, "START")
workflow.add_edge("START", "supervisor")

graph = workflow.compile()
graph.get_graph().draw_png("multi-agent_workflow.png")
init_state = {"messages": [HumanMessage(content="I have a hypothesis that claims the number of deaths increases over "
                                                "the years. Can you prove it whit using necessary tests?")],
              "next": "START"
              }

for s in graph.stream(init_state):
    if "__end__" not in s:
        print(s)
        print("----")
