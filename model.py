from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory  # still works for now
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import chainlit as cl

# Prompt Template
assistant_template = """
You are an Anatomy assistant chatbot named "Scoopsie". Your expertise is 
exclusively in providing information and advice about anything related to 
medical Anatomy book topics. You do not provide information outside of this 
scope. If a question is not about Anatomy, respond with, "I specialize only in Anatomy related queries."

Chat History: {chat_history}
Question: {question}
Answer:"""

assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=assistant_template
)

# Chainlit event on chat start
@cl.on_chat_start
def start_chat():
    llm = LlamaCpp(
        model_path="unsloth.Q4_K_M.gguf",
        temperature=0.7,
        n_ctx=2048,
        verbose=True
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_len=50
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=assistant_prompt_template,
        memory=memory
    )

    cl.user_session.set("llm_chain", llm_chain)

# Chainlit event on user message
@cl.on_message
async def handle_message(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")

    response = await llm_chain.acall(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    await cl.Message(content=response["text"]).send()

