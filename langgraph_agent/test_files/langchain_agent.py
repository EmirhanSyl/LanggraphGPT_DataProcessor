from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

db = [{
    'username': 'ahmet',
    'password': '123',
}]

translator = ChatOllama(
    model="aya",
    temperature=0,
)

translator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a translator that translates {input_language} to {output_language}. Do not add anything to the message just translate it. Following is the user query: ",
        ),
        ("human", "{input}"),
    ]
)

chain = translator_prompt | translator
translated_msg = chain.invoke(
    {
        "input_language": "Turkish",
        "output_language": "English",
        "input": "kullanıcı adı 'ahmet' ve şifresi '123' olan bir kullanıcının varlığını doğrular mısın?",
    }
)

print(translated_msg.content)


def validate_user(username: str, password: str) -> bool:
    """Validate user using username and password.

    Args:
        username: (int) the username of the user.
        password: (str) user's password.
    """
    for entry in db:
        if entry['username'] == username and entry['password'] == password:
            return True
    return False


llm = ChatOllama(
    model="llama3-groq-tool-use",
    temperature=0,
).bind_tools([validate_user])

llm_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a password validators for the users. Validate Passwords and if user is valid, say 'holla!'",
        ),
        ("human", "{input}"),
    ]
)

chain = llm_prompt | llm

result = chain.invoke(
    {
        "input": translated_msg.content,
    }
)
print(result.tool_calls)
print(result)