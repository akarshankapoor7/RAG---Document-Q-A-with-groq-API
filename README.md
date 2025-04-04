RAG Document Q&A with Streamlit, LangChain & Llama3
This project is a Retrieval-Augmented Generation (RAG) Document Q&A application built with Streamlit, LangChain, and Groq’s Llama3 model. It allows users to query information from PDF documents stored in a specified directory (dox).

The system processes PDFs using PyPDFDirectoryLoader, splits them into chunks with RecursiveCharacterTextSplitter, and embeds them into a FAISS vector store using HuggingFace’s all-MiniLM-L6-v2 embeddings. When a user inputs a query, the app retrieves relevant document chunks via a retrieval chain and generates answers using the Llama3-8b-8192 model, displaying both the response and similar document excerpts.

Features
Loads PDF documents from the dox directory.

Splits documents into chunks for better retrieval.

Stores document embeddings in a FAISS vector database.

Uses retrieval chains to fetch relevant document sections.

Generates answers using Groq’s Llama3 model.

Displays relevant document excerpts for context.

Usage
Add PDF documents to the dox directory.

Click the "Submit" button to process and store document embeddings.

Enter a query in the text input box.

View the generated answer along with similar document excerpts.

Dependencies
