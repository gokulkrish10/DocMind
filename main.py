import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DocuMind AI - Intelligent Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .capability-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
    }
    
    .question-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .answer-box {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-top: 1rem;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧠 DocuMind AI</h1>
    <h3>Intelligent Document Question & Answer Assistant</h3>
    <p>Transform your PDFs into conversational knowledge with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# Constants
UPLOAD_FILES = "pdfs"
INDEX_FOLDER = "pdf_faiss_index"
os.makedirs(UPLOAD_FILES, exist_ok=True)

# Initialize session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "embedding" not in st.session_state:
    st.session_state.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
if "db" not in st.session_state:
    st.session_state.db = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Sidebar with tool information
with st.sidebar:
    st.markdown("## 📖 About DocuMind AI")
    
    st.markdown("""
    **DocuMind AI** is your intelligent document companion that transforms static PDFs into interactive knowledge bases.
    
    ### 🎯 What This Tool Does:
    - **Reads & Understands** your PDF documents using advanced AI
    - **Answers Questions** in natural language about your content
    - **Provides Context** with relevant information from your documents
    - **Saves Time** by instantly finding information across multiple PDFs
    """)
    
    st.markdown("### 🚀 Key Capabilities:")
    
    capabilities = [
        "📄 Multi-PDF Processing",
        "🔍 Intelligent Search",
        "💬 Natural Language Q&A",
        "🎯 Contextual Responses",
        "⚡ Fast Information Retrieval",
        "🔒 Secure Local Processing"
    ]
    
    for capability in capabilities:
        st.markdown(f"<div class='capability-box'>{capability}</div>", unsafe_allow_html=True)
    
    st.markdown("### 📋 How to Use:")
    st.markdown("""
    1. **Upload** your PDF documents
    2. **Wait** for processing to complete
    3. **Ask** questions about your content
    4. **Get** instant, accurate answers
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📁 Upload Documents")
    
    with st.container():
        
        
        upload_files = st.file_uploader(
            "Choose your PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to create your knowledge base"
        )
        
        if upload_files:
            st.success(f"✅ {len(upload_files)} file(s) selected!")
            
            # Show uploaded files
            st.markdown("**Selected Files:**")
            for file in upload_files:
                st.markdown(f"• {file.name} ({file.size/1024:.1f} KB)")
        
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 🔧 Processing Status")
    
    if st.session_state.pdf_processed:
        st.success("🎉 Documents processed successfully!")
        st.markdown(f"**Processed Files:** {len(st.session_state.processed_files)}")
        
        # Show processed files
        with st.expander("View Processed Documents"):
            for file in st.session_state.processed_files:
                st.markdown(f"✅ {file}")
        
        # Reset button
        if st.button("🔄 Process New Documents", type="secondary"):
            st.session_state.pdf_processed = False
            st.session_state.db = None
            st.session_state.processed_files = []
            st.rerun()
    else:
        st.info("📝 Upload PDF files and click 'Process Documents' to begin")

# Process documents
if upload_files and not st.session_state.pdf_processed:
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        
        # Save uploaded files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("💾 Saving uploaded files...")
        for i, file in enumerate(upload_files):
            file_path = os.path.join(UPLOAD_FILES, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            progress_bar.progress((i + 1) / len(upload_files) / 3)
        
        status_text.text("📖 Loading and processing documents...")
        documents = []
        uploaded_files = [file.name for file in upload_files]
        
        for i, file in enumerate(os.listdir(UPLOAD_FILES)):
            if file in uploaded_files:
                loader = PyPDFLoader(os.path.join(UPLOAD_FILES, file))
                documents.extend(loader.load())
                st.session_state.processed_files.append(file)
            progress_value = (1/3) + (i + 1) / len(upload_files) / 3
            progress_bar.progress(min(progress_value, 1.0))
        
        status_text.text("✂️ Splitting documents into chunks...")
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        progress_bar.progress(2/3)
        
        status_text.text("🧠 Creating AI embeddings...")
        db = FAISS.from_documents(docs, st.session_state.embedding)
        db.save_local(INDEX_FOLDER)
        
        st.session_state.db = db
        st.session_state.pdf_processed = True
        progress_bar.progress(1.0)
        status_text.text("🎉 Processing completed successfully!")
        
        time.sleep(1)
        st.rerun()

# Question-Answer Section
if st.session_state.pdf_processed:
    st.markdown("---")
    st.markdown("## 💬 Ask Questions About Your Documents")
    
    #st.markdown('<div class="question-section">', unsafe_allow_html=True)
    
    # Example questions
    st.markdown("### 💡 Example Questions You Can Ask:")
    example_questions = [
        "What is the main conclusion of the research?",
        "Summarize the methodology used in the study",
        "What are the key findings mentioned?",
        "Who are the authors of this document?",
        "What problems does this research address?"
    ]
    
    cols = st.columns(len(example_questions))
    for i, question in enumerate(example_questions):
        with cols[i]:
            if st.button(f"❓ {question}", key=f"example_{i}", help="Click to use this question"):
                st.session_state.example_question = question
    
    # Question input
    question = st.text_input(
        "🔍 Enter your question:",
        value=getattr(st.session_state, 'example_question', ''),
        placeholder="Ask anything about your uploaded documents...",
        help="Type your question in natural language"
    )
    
    if question:
        if st.session_state.db is None:
            # Load DB from disk
            with st.spinner("Loading knowledge base..."):
                st.session_state.db = FAISS.load_local(
                    INDEX_FOLDER,
                    st.session_state.embedding,
                    allow_dangerous_deserialization=True
                )
        
        # Search for relevant documents
        with st.spinner("🔍 Searching through your documents..."):
            docs = st.session_state.db.similarity_search(question, k=3)
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=os.getenv("OPEN_API_KEY"),
            temperature=0
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")
        
        # Generate answer
        with st.spinner("🧠 AI is thinking and generating your answer..."):
            answer = chain.run(input_documents=docs, question=question)
        
        # Display answer
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Answer:")
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show sources
        with st.expander("📚 View Source Documents"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:**")
                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🧠 <strong>DocuMind AI</strong> - Powered by Advanced Language Models & Vector Search</p>
    <p>Transform your documents into intelligent, searchable knowledge bases</p>
</div>
""", unsafe_allow_html=True)