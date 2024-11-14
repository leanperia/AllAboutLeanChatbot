""" Lean Chatbot
@author: Lean Peria
@email: leanlouiel@gmail.com

"""

import gradio as gr
import random
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

"""
##### Failed attempt to use Donut for OCR on scanned images of documents #######
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pdf2image
import torch
from langchain_core.documents import Document 

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
def load_pdf_with_donut(pdf_path):
    # Convert PDF to images
    images = pdf2image.convert_from_path(pdf_path)
    docs = []
    for image in images:
        # Convert image to RGB
        image = image.convert("RGB")
        # Process the image
        pixel_values = processor(image, return_tensors="pt").pixel_values
        # Generate text from the image
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Create a Document object
        docs.append(Document(page_content=generated_text))
    return docs
"""
    
def new_thread():
    random_string = f"{random.random():.8f}"[2:]
    config = {"configurable": {"thread_id": random_string}}
    return config

current_config = new_thread()
chat_configs = [current_config]
memory = MemorySaver()

def initialize_LLM(progress=gr.Progress()):
    hfembed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={'normalize_embeddings': False}
    )
    llm = ChatAnthropic(model_name="claude-3-sonnet-20240229")

    file_path = "leandocs/writeups.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    writeups_db = Chroma.from_documents(
        documents=splits, embedding=hfembed
    )
    retriever=writeups_db.as_retriever()
    writeups_retriever = create_retriever_tool(
        retriever,
        "writeups_retriever",
        """Searches and returns excerpts from writeups written by Lean Louiel Peria. These writeups were
            submissions for various English courses in college back in 2012-2014. They showcase his writing style
            and throught processes as a young college student more than 10 years ago.
        """,
        )
    
    file_path = "leandocs/resume.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    resume_db = Chroma.from_documents(
        documents=splits, embedding=hfembed
    )
    retriever = resume_db.as_retriever()
    resume_retriever = create_retriever_tool(
        retriever,
        "resume_retriever",
        """Searches and returns information from Lean Louiel Peria's up-to-date resume. It summarizes his
            professional experience, education, and skills.
        """,
    )

    file_path = "leandocs/disc.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    disc_db = Chroma.from_documents(
        documents=splits, embedding=hfembed
    )
    retriever = disc_db.as_retriever()
    disc_retriever = create_retriever_tool(
        retriever,
        "disc_retriever",
        """Searches and returns information from the result of the DISC personality test taken
            by Lean Louiel Peria."""
    )

    search = TavilySearchResults(max_results=2)
    tools = [writeups_retriever, resume_retriever, disc_retriever, search]

    system_prompt = """You are an AI chatbot with knowledge about Lean Louiel Peria. Use the various retriever tools available to you to answer questions about him or his work. 
                       If you cannot answer the question from the source documents, use the Tavily search tool to look up information on the internet.

                       If the user asks what kinds of questions they can ask about Lean, you can suggest the following:
                          - Ask about his professional experience or education
                          - Ask about his personality
                          - Ask about his English writeups in college
                    """
    agent = create_react_agent(llm, tools, checkpointer=memory, state_modifier=system_prompt)
    return agent, "Chatbot is ready!"
    
def reset_chatbot():
    global current_config, chat_configs
    current_config = new_thread()
    chat_configs.append(current_config)
    return None

def conversation(qa_chain, message, history):
    response = qa_chain.invoke({"messages": [HumanMessage(content=message)]}, config=current_config)
    response_answer = response["messages"][-1].content
    history.append(
        {'role': 'user', 'content': message}
    )
    history.append(
        {'role': 'assistant', 'content': response_answer}
    )
    return qa_chain, gr.update(value=""), history

def demo():
    with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue = "sky")) as demo:
        qa_chain = gr.State()
        gr.HTML("<center><h1>All about Lean!</h1><center>")
        gr.HTML("<center>Hello! I am an AI agent designed to answer questions about Lean Louiel Peria.<center>")

        with gr.Row():
            with gr.Column(scale = 86):
                with gr.Row():
                    qachain_btn = gr.Button("Initialize Question Answering Chatbot")
                with gr.Row():
                        llm_progress = gr.Textbox(value="Not initialized", show_label=False) 
                gr.Markdown("""
                    I use the following sources:
                    - Resume
                    - Writeups from various English courses in college
                    - DISC Personality Test results
                """)
            with gr.Column(scale = 200):
                chatbot = gr.Chatbot(height=400, type="messages")               
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a question", container=True)
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.ClearButton([msg, chatbot], value="Clear")
            
        qachain_btn.click(initialize_LLM, \
            inputs=None, \
            outputs=[qa_chain, llm_progress]).then(lambda:None, 
            inputs=None, \
            outputs = [chatbot], \
            queue=False)

        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot],\
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot],\
            queue=False)
        clear_btn.click(reset_chatbot, \
            inputs=None, \
            outputs=[chatbot], \
            queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()