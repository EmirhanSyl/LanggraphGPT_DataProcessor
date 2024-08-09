from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

translator = ChatOllama(
    model="aya",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful translator that translates {input_language} to {output_language}.",
        ),
        ("human", "Translate this: {input}"),
    ]
)

chain = prompt | translator
aimessage = chain.invoke(
    {
        "input_language": "Turkish",
        "output_language": "English",
        "input": "Veri setimde bulunan flight_times isimli kolonu datetime türüne dönüştürüp bu kolona bir time shifting işlemi uygulamak istiyorum",
    }
)

print(aimessage)

