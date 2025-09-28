import uuid
import sqlite3
from datetime import datetime
import os
import json
import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader


# ================== CUSTOM CSS ==================
st.markdown("""
<style>
.stChatInputContainer {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    z-index: 1001 !important;
    background: #fff;
    border-top: 1px solid #d1d5db;
    padding: 1rem 2rem !important;
    box-shadow: 0 -2px 12px rgba(0,0,0,0.06);
}
.block-container {
    padding-bottom: 7rem !important;
}
.chat-messages {
    max-height: calc(100vh - 10rem);
    overflow-y: auto;
    padding: 1rem 2rem;
}
.user-message {
    display: flex;
    justify-content: flex-end;
    margin-left: 2rem;
}
.user-message .markdown {
    background: #10a37f;
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 18px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
.assistant-message {
    display: flex;
    justify-content: flex-start;
    margin-right: 2rem;
}
.assistant-message .markdown {
    background: #ffffff;
    color: #333333;
    padding: 1rem 1.25rem;
    border-radius: 18px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
    border: 1px solid #e5e5e5;
}
</style>
""", unsafe_allow_html=True)


# ================== DATABASE ==================
def init_database():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT,
            session_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            message_type TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
    ''')
    conn.commit()
    conn.close()


def save_message(session_id, message_type, content):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    message_text = getattr(content, "content", str(content))   # Sirf text save karega
    cursor.execute('''
        INSERT INTO chat_messages (session_id, message_type, content)
        VALUES (?, ?, ?)
    ''', (session_id, message_type, message_text))
    conn.commit()
    conn.close()


def create_new_session(user_id, session_name=None):
    session_id = str(uuid.uuid4())
    if not session_name:
        session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_sessions (user_id, session_id, session_name)
        VALUES (?, ?, ?)
    ''', (user_id, session_id, session_name))
    conn.commit()
    conn.close()
    return session_id


def get_session_messages(session_id):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT message_type, content, timestamp
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    ''', (session_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages


# ================== LLM & PROMPT ==================
system_prompt = """
You are Mindful AI, a compassionate and supportive mental health companion.
- Be empathetic, calm, and non-judgmental.
- Give short, practical, and encouraging advice.
- If user asks technical/programming, answer step by step with clarity.
- If user asks personal/emotional, focus on positivity and support.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])


def initialize_llm():
    return ChatGroq(
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY", "api here"),
        model_name="llama-3.3-70b-versatile"
    )


# ================== RAG (Knowledge Base) ==================
def init_vector_db():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    return vectordb


def build_qa_chain(llm, vectordb):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff"
    )


def load_pdf_to_db(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    vectordb = init_vector_db()
    vectordb.add_documents(docs)
    return vectordb


# ================== STREAMLIT APP ==================
init_database()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate mental health support companion. How can I support you today?"}
    ]

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:16]

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = create_new_session(st.session_state.user_id)

if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()


# Chat UI
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><div class="markdown">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><div class="markdown">{message["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# User Input
prompt = st.chat_input("Type your message here...", key="static_chat_input")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.current_session_id, "user", prompt)

    # Default LLM reply with system prompt
    formatted = prompt_template.format_messages(input=prompt)
    assistant_reply = st.session_state.llm.invoke(formatted)

    # OPTIONAL: If you want knowledge base answers
    vectordb = init_vector_db()
    qa_chain = build_qa_chain(st.session_state.llm, vectordb)
    assistant_reply = qa_chain.run(prompt)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    save_message(st.session_state.current_session_id, "assistant", assistant_reply)

    st.rerun()
