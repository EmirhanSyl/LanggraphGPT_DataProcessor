import json
import asyncio
from langgraph_agent.tools.tools import pdf

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph_agent.agent_state import MessageTypes
from app import App
from langgraph_agent.tools.tools import set_dataset

app = App()
global task_list
task_part_list = []
current_task = 0


async def init_task_list():
    # Create the TaskList
    global task_list
    task_list = cl.TaskList()
    task_list.status = "Çalışıyor..."

    task1 = cl.Task(title="Veriseti analizi", status=cl.TaskStatus.RUNNING)
    task2 = cl.Task(title="Veriseti önişleme", status=cl.TaskStatus.READY)

    await task_list.add_task(task1)
    await task_list.add_task(task2)

    task_part_list.append(task1)
    task_part_list.append(task2)

    # Update the task list in the interface
    await task_list.send()
    await asyncio.sleep(0.5)


# Callback function to update a specific task's status
async def update_task_status(task, new_status):
    task.status = new_status  # Update task status
    await task_list.send()
    await asyncio.sleep(0.5)


async def add_task(title, statues):
    new_task = cl.Task(title=title, status=statues)

    task_part_list.append(new_task)
    await task_list.add_task(new_task)

    await task_list.send()
    await asyncio.sleep(0.5)


async def inform_about_preprocessing():
    response = app.stream_app()

    image = cl.Image(path="./public/images/missing_handling_graph.jpg", name="handle_missing", display="inline", size="large")
    await cl.Message(
        content="Missing Handling Strategy Graph",
        elements=[image],
    ).send()

 
    await update_task_status(task_part_list[1], cl.TaskStatus.READY)  # Magic Number! Change it
    res = await cl.AskActionMessage(
        content=response.content,
        actions=[
            cl.Action(name="continue", value="continue", label="✅ Continue"),
            cl.Action(name="cancel", value="cancel", label="❌ Cancel"),
        ],
        timeout=99999
    ).send()

    if res and res.get("value") == "continue":
        # Update Frontend
        await update_task_status(task_part_list[1], cl.TaskStatus.RUNNING)
        await add_task("Eksik Değerlerin Giderilmesi", cl.TaskStatus.RUNNING)
        await add_task("Aykırı Değerlerin Giderilmesi", cl.TaskStatus.READY)
        await add_task("Önişleme Sonuçları", cl.TaskStatus.READY)

        # Update Graph
        await preprocess_results()
    elif res and res.get("value") == "cancel":
        # Update Frontend
        await cl.Message(content="Preprocessing skipped. Ask me anything about your dataset!").send()
        task_list.status = "İptal Edildi."
        await update_task_status(task_part_list[1], cl.TaskStatus.FAILED)

        # Update Graph
        snapshot = app.app_runnable.get_state(app.thread)
        snapshot.values['messages'] += [HumanMessage(content="I denied to preprocessing steps.")]
        app.app_runnable.update_state(app.thread, snapshot.values, as_node="ask_to_model")


async def preprocess_results():
    response = app.stream_app()  # Run Handle Missing

    snapshot = app.app_runnable.get_state(app.thread)

    tool_message = snapshot.values["messages"][-3]  # Tool Message Index
    tool_message_json = json.loads(tool_message.content)
    await cl.Message(content=tool_message_json).send()
    await asyncio.sleep(0.5)
    await cl.Message(content=response.content).send()

    await update_task_status(task_part_list[2], cl.TaskStatus.DONE)
    await update_task_status(task_part_list[3], cl.TaskStatus.RUNNING)

    response = app.stream_app()  # Run Handle Outlier
    snapshot = app.app_runnable.get_state(app.thread)

    tool_message = snapshot.values["messages"][-3]  # Tool Message Index
    tool_message_json = json.loads(tool_message.content)
    await cl.Message(tool_message_json).send()
    await cl.Message(content=response.content).send()

    await update_task_status(task_part_list[3], cl.TaskStatus.DONE)
    await update_task_status(task_part_list[4], cl.TaskStatus.RUNNING)

    response = app.stream_app()  # Run End Of Preprocess

    await update_task_status(task_part_list[4], cl.TaskStatus.DONE)
    task_list.status = "Tamamlandı"
    await update_task_status(task_part_list[1], cl.TaskStatus.DONE)
    await cl.Message(content=response.content).send()

    print(snapshot)
    pdf()
    # Sending a pdf with the local file path
    elements = [
        cl.Pdf(name="pdf1", display="inline", path="./output.pdf")
    ]
    # Reminder: The name of the pdf must be in the content of the message
    await cl.Message(content="Here is the dataset after preprocessing.", elements=elements).send()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("runnable", app.app_runnable)
    response = app.stream_app({"messages": [HumanMessage(content="")]})

    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content=response.content, accept=["text/csv"], max_files=1, timeout=999999
        ).send()

    await init_task_list()

    dataset = files[0]
    set_dataset(dataset.path)
    app.stream_app()

    snapshot = app.app_runnable.get_state(app.thread)
    print(snapshot.values)

    tool_message = snapshot.values["messages"][-3]  # Tool Message Index
    tool_message_json = json.loads(tool_message.content)
    await cl.Message(content=tool_message_json).send()

    result_message = snapshot.values["messages"][-1]
    await cl.Message(content=result_message.content).send()

    await update_task_status(task_part_list[0], cl.TaskStatus.DONE)
    await update_task_status(task_part_list[1], cl.TaskStatus.RUNNING)
    await inform_about_preprocessing()


@cl.on_message
async def on_message(message: cl.Message):
    human_message = HumanMessage(content=message.content)

    snapshot = app.app_runnable.get_state(app.thread)
    snapshot.values['messages'] += [human_message]

    app.app_runnable.update_state(config=app.thread, values=snapshot.values, as_node="ask_to_model")

    response = app.stream_app()

    message_type = snapshot.values["last_message_type"]
    if message_type == MessageTypes.CHAT:
        await cl.Message(content=response.content).send()
    elif message_type == MessageTypes.VERIFICATION:
        actions = [
            cl.Action(name="approve_tool_use", value="approve", description="approve"),
            cl.Action(name="deny_tool_use", value="deny", description="deny"),
        ]
        await cl.Message(content=response.content, actions=actions).send()
    snapshot = app.app_runnable.get_state(app.thread)
    print(snapshot.values)

