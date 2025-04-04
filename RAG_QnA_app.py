import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# API Keys
api_key = os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['HF_TOKEN'] = "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"  # Secure token retrieval

# Define PDF Directory (using forward slashes)
PDF_DIRECTORY = "C:/Users/Akarshan Kapoor/Documents/Data Science/projx gen AI/RAG Document Q&A/dox"

# Initialize LLM
llm = ChatGroq(api_key=api_key, model="llama3-8b-8192", temperature=0.6)

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader(PDF_DIRECTORY)  # Use the defined variable
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.title("RAG Document Q&A With Groq And Llama3")

# Button to create embeddings
if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("Vector Database is ready")

# User query input
user_prompt = st.text_input("Enter your query from the research paper")

# Process query only if embeddings exist and a query is provided
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    response_time = time.process_time() - start

    st.write(f"Response time: {response_time:.2f} seconds")
    st.write("Answer:", response.get('answer', "No answer found"))

    # Document similarity search expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get('context', [])):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write('------------------------')
elif user_prompt and "vectors" not in st.session_state:
    st.warning("Please create the vector database first by clicking 'Document Embedding'.")