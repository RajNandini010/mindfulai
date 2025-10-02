import uuid
import sqlite3
from datetime import datetime
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

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

def create_new_session(user_id, session_name="New Chat"):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_sessions (user_id, session_id, session_name) VALUES (?, ?, ?)",
        (user_id, session_id, session_name)
    )
    conn.commit()
    conn.close()
    return session_id

def save_message(session_id, message_type, content):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    message_text = getattr(content, "content", str(content))
    cursor.execute('''
        INSERT INTO chat_messages (session_id, message_type, content)
        VALUES (?, ?, ?)
    ''', (session_id, message_type, message_text))
    conn.commit()
    conn.close()

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
    return [{"role": "user" if m[0]=="user" else "assistant", "content": m[1]} for m in messages]

# ================== LLM ==================
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
        groq_api_key=os.getenv("GROQ_API_KEY", "api_here"),
        model_name="llama-3.1-8b-instant"
    )

# ================== AUTO SESSION NAMING ==================
def generate_session_name(llm, first_message: str) -> str:
    instruction = f"Summarize this message into 3-4 words for a chat title:\n\n{first_message}"
    response = llm.invoke(instruction)
    title = response.content.strip()
    return f"Chat about {title}"

def update_session_name_with_llm(session_id, first_message, llm):
    session_name = generate_session_name(llm, first_message)
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET session_name = ? WHERE session_id = ?",
        (session_name, session_id)
    )
    conn.commit()
    conn.close()

# ================== STREAMLIT APP ==================
init_database()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
    ]

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:16]

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = create_new_session(st.session_state.user_id)

if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("💬 Mindful AI")
    st.write("Your conversations")
    
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, session_name FROM chat_sessions ORDER BY created_at DESC")
    sessions = cursor.fetchall()
    conn.close()

    for s_id, s_name in sessions:
        if st.button(s_name, key=s_id):
            st.session_state.current_session_id = s_id
            st.session_state.messages = get_session_messages(s_id)
            st.rerun()

# ========== CHAT UI ==========
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><div class="markdown">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><div class="markdown">{message["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== USER INPUT ==========
prompt = st.chat_input("Type your message here...")

if prompt:
    # save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.current_session_id, "user", prompt)

    # update session name if it's still "New Chat"
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT session_name FROM chat_sessions WHERE session_id=?", (st.session_state.current_session_id,))
    current_name = cursor.fetchone()[0]
    conn.close()

    if current_name == "New Chat":
        update_session_name_with_llm(st.session_state.current_session_id, prompt, st.session_state.llm)

    # assistant reply
    formatted = prompt_template.format_messages(input=prompt)
    assistant_reply = st.session_state.llm.invoke(formatted)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply.content})
    save_message(st.session_state.current_session_id, "assistant", assistant_reply.content)

    st.rerun()

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
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
    padding: 0.8rem 1.2rem;
    border-radius: 20px 20px 5px 20px;
    max-width: 70%;
}
.assistant-message {
    display: flex;
    justify-content: flex-start;
    margin-right: 2rem;
}
.assistant-message .markdown {
    background: #f5f5f5;
    color: #222;
    padding: 0.8rem 1.2rem;
    border-radius: 20px 20px 20px 5px;
    max-width: 70%;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)
