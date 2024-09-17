import asyncio

from langgraph_agent.models.aya_model import ModelAya
from langchain_core.messages import HumanMessage, SystemMessage


class Translator:
    def __init__(self):
        self.translator = ModelAya().llm

    async def translate(self, text):
        system_prompt = SystemMessage(content="You are a translator.Translate everything that user gives you to Turkish")
        user_message = HumanMessage(content=text)
        response = self.translator.invoke([system_prompt, user_message]).content

        while response is None:
            await asyncio.sleep(0.2)

        return response
