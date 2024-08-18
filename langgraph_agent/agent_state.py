from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from enum import Enum


class MessageTypes(Enum):
    CHAT = 0
    TOOL_USE = 1
    VERIFICATION = 2


class State(TypedDict):
    messages: Annotated[list, add_messages]
    last_message_type: MessageTypes
    last_called_tool: dict

