import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    pass


@cl.on_message
async def on_message(message: cl.Message):
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Merhaba! Ben AI asistanınız Beeg. Veri analizi platformumuza hoş geldiniz! Veriyle ilgili herhangi"
                    " bir görevinize yardımcı olmak için buradayım. Analiz etmek istediğiniz bir CSV veri kümesini bana"
                    " sağlayabilir misiniz?", accept=["text/csv"], max_files=1, timeout=999999
        ).send()
