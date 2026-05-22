import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import tempfile

st.set_page_config(
    page_title="SreeAI PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 SreeAI PDF Chatbot")
st.subheader("💬 Conversational RAG AI Assistant")

with st.sidebar:
    st.title("About")
    st.info("""
     - Upload a PDF and ask questions from it.
     - This chatbot answers only from the uploaded PDF.
     - This chatbot doesn't support multiple PDFs.
     - Uploading a new PDF, removing the PDF, or clicking clear chat will reset the conversation memory.
    """)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.vectordb = None
        st.session_state.uploaded_file = None
        st.session_state.uploader_key += 1
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )

file = st.file_uploader("**Upload PDF**", 
                        type="pdf", 
                        accept_multiple_files=False,
                        key=st.session_state.uploader_key)

if file is not None:
    if file.name != st.session_state.uploaded_file:
        st.session_state.messages = []
        st.session_state.vectordb = None
        st.session_state.uploaded_file = file.name

if file:     
    if st.session_state.vectordb is None:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                file_path = temp_file.name
                    
            documents = PyPDFLoader(file_path).load()
        
            textsplitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=30
                        )
                
            chunks = textsplitter.split_documents(documents)
                
            embeddings = HuggingFaceEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
                
            st.session_state.vectordb = FAISS.from_documents(chunks, embeddings)
            st.success("✅ PDF processed successfully")
        
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
    
    query = st.chat_input("Ask anything from the PDF...")

    if query and query.strip():
        with st.chat_message("user", avatar="🧑"):
            st.write(query)
            
        st.session_state.messages.append({
            "role":"user",
            "content":query
        })
    
        docs = st.session_state.vectordb.max_marginal_relevance_search(query, k=8, fetch_k=20)

        context = "\n\n".join([
            f"Page: {doc.metadata.get('page')+1}\n"
            f"Content: {doc.page_content}" for doc in docs
        ])
        
        chat_history = "\n".join([
            f"{msg['role']}:{msg['content']}" for msg in st.session_state.messages
        ])
    
        prompt = f"""
        You are a helpful PDF assistant.

        IMPORTANT RULES:

        - Answer ONLY from the uploaded PDF.
        - Do NOT make up information.
        - If answer is unavailable, say:
          "I could not find that information in the PDF."

        conversation history: {chat_history}
    
        context: {context}
    
        question: {query}
        """
    
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            response = ""
            for chunk in st.session_state.llm.stream(prompt):
                response += chunk.content
                placeholder.markdown(response+"▌")
            placeholder.markdown(response)
    
        st.session_state.messages.append({
            "role":"assistant",
            "content":response
        })