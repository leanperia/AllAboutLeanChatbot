---
title: LeanChatbot
emoji: 🐠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
license: mit
short_description: This chatbot answers questions about Lean Louiel Peria
---

# Lean's AI Chatbot

This is a gradio app of a simple chatbot that uses underlying RAG (Retrieval-Augmented Generation) mechanisms to answer questions about Lean based on a set of documents he provided. 

You can run this on HuggingFace spaces or locally. 

First, make sure to click the button to initiate the chatbot. This will load the documents about Lean.
After that you can start conversing with the chatbot. 

For local installation:
```
pip install -r requirements.txt
```

Also, make sure you have the variables below in a .env file in the root directory. As such, you should also have access and credits to anthropic, huggingface and tavily APIs. 
```
ANTHROPIC_API_KEY
HUGGINGFACEHUB_API_TOKEN
TAVILY_API_KEY
```

# Design

The underlying architecture for this chatbot is a prebuilt ReAct agent from langgraph. I fed it several tools:
- 3 retriever tools for a vector store of embeddings for each of the following: the English writeups, the resume, and the personality test
- a Tavily search tool 

I believed I could much better guide the LLM to decide where to find the desired information if I partitioned my documents into separate topics, and take advantage of the retriever descriptor prompt. 

I also think it is a good user experience if they can also ask questions that are searchable in the internet. For example, they can ask about Proper nouns (ie University of the Philippines) mentioned in my source documents.

Since all of the documents are in PDF format, it was just simple to extract the data using PyPDFLoader. 

I opted for HuggingFace for the embeddings model. I tried to force HuggingFace Llama for the chat model but the free tier 4096 tokens per call is simply not enough for a RAG. So I opted to use my remaining credits in Anthropic. 

# Evaluation
Open any of the source documents and ask specific questions about them. You can also ask follow-up questions, or even ask more encompassing or summarizing questions as the LLM can handle it.

It is possible to create evaluation datasets running tests via API to more objectively measure the performance of this RAG, however it will take too much time for me.

# Reference
I would like to acknowledge the HuggingFace space linked below that I used as reference for building my gradio app:
`https://huggingface.co/spaces/MuntasirHossain/RAG-PDF-Chatbot`


--

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
