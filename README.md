📄 RAG Document Q&A with Streamlit, LangChain & Llama3
🔍 Overview
This project is a Retrieval-Augmented Generation (RAG) Document Q&A application built with:
✅ Streamlit (UI)
✅ LangChain (Document Processing & Retrieval)
✅ Groq’s Llama3 Model (LLM for Answer Generation)

It enables users to query PDF documents stored in a directory (dox) and retrieve relevant information efficiently.

🚀 Features
✔ PDF Document Processing: Uses PyPDFDirectoryLoader to load PDFs
✔ Text Splitting: Splits documents into chunks via RecursiveCharacterTextSplitter
✔ Vector Storage: Stores embeddings in a FAISS vector database
✔ Query Processing: Retrieves relevant chunks using retrieval chains
✔ LLM Answer Generation: Uses Llama3-8b-8192 for response generation
✔ Document Similarity Search: Displays similar document excerpts
✔ Response Time Tracking

🛠️ Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
2️⃣ Create a Virtual Environment
bash
Copy
Edit
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate  # On Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Set Up Environment Variables
Create a .env file in the project root and add:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
HF_TOKEN=your_huggingface_api_token
5️⃣ Run the Application
bash
Copy
Edit
streamlit run app.py
📂 Project Structure
bash
Copy
Edit
📂 RAG Document Q&A
│── 📂 dox                  # Directory for storing PDF documents
│── 📜 app.py               # Main application script
│── 📜 requirements.txt      # Dependencies
│── 📜 .env                  # Environment variables (not shared)
│── 📜 README.md             # Project Documentation
📝 Usage
1️⃣ Upload PDF files into the dox directory
2️⃣ Click the "Submit" button to process and store document embeddings
3️⃣ Enter a query in the text input box
4️⃣ View the answer & similar document excerpts

📌 Dependencies
streamlit

langchain

faiss-cpu

sentence-transformers

python-dotenv

pypdf

groq

huggingface_hub

🔗 References
LangChain Documentation: https://python.langchain.com

Groq API: https://groq.com

FAISS: https://github.com/facebookresearch/faiss

