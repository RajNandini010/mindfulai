import uuid
import sqlite3
from datetime import datetime
import os
import re
import bcrypt
import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ================== EMAIL VALIDATION ==================
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[a-zA-Z\d-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# ================== DATABASE ==================
def init_database():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT,
            session_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            message_type TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    ''')
    conn.commit()
    conn.close()

def register_user(username, email, password):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                      (username, email, hashed_pw))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Email already exists"
    conn.close()
    return True, "Registration successful"

def login_user(email, password):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode(), user[2]):
        return True, user[0], user[1]
    return False, None, None

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
    
    if not messages:
        return []
    
    return [{"role": "user" if m[0]=="user" else "assistant", "content": m[1]} for m in messages]

# ================== LLM AND PROMPT ==================
system_prompt = """
You are Mindful AI, a compassionate and supportive mental health companion.
- Be empathetic, calm, and non-judgmental.
- Give short, practical, and encouraging advice.
- If user asks technical/programming, answer step by step with clarity.
- If user asks personal/emotional, focus on positivity and support.
- Remember the conversation context and refer to previous messages when relevant.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def initialize_llm():
    return ChatGroq(
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY", "your_api_key_here"),
        model_name="llama-3.1-8b-instant"
    )

def get_chat_history_for_llm(messages):
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

# ================== SUGGESTION GENERATION ==================
def get_dynamic_suggestions(llm, messages):
    suggestion_prompt = (
        "Based on our conversation so far, suggest 3 short helpful questions or topics "
        "the user can ask next, separated by | symbol."
    )
    history = get_chat_history_for_llm(messages)
    formatted = prompt_template.invoke({
        "chat_history": history,
        "input": suggestion_prompt
    })
    response = llm.invoke(formatted)
    suggestions = [s.strip() for s in response.content.split('|') if s.strip()]
    return suggestions[:5]

# ================== HANDLE SUGGESTION CLICK ==================
def handle_suggestion(suggestion_text):
    st.session_state.messages.append({"role": "user", "content": suggestion_text})
    save_message(st.session_state.current_session_id, "user", suggestion_text)

    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT session_name FROM chat_sessions WHERE session_id=?", (st.session_state.current_session_id,))
    result = cursor.fetchone()
    conn.close()

    current_name = result[0] if result and result[0] else "New Chat"
    if current_name == "New Chat":
        update_session_name_with_llm(st.session_state.current_session_id, suggestion_text, st.session_state.llm)
    
    chat_history = get_chat_history_for_llm(st.session_state.messages[:-1])
    formatted = prompt_template.invoke({
        "chat_history": chat_history,
        "input": suggestion_text
    })
    assistant_reply = st.session_state.llm.invoke(formatted)
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply.content})
    save_message(st.session_state.current_session_id, "assistant", assistant_reply.content)
    st.session_state.show_suggestions = False

# ================== STREAMLIT APP ==================
init_database()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# --- Login/Register UI
if not st.session_state.logged_in:
    st.title("🔒 Login or Register")
    option = st.radio("Choose action", ("Login", "Register"))
    
    if option == "Register":
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            if not username:
                st.error("Please enter a username")
            elif not email:
                st.error("Please enter an email")
            elif not password:
                st.error("Please enter a password")
            else:
                success, msg = register_user(username, email, password)
                if success:
                    st.success(msg)
                    st.info("Please login with your credentials")
                else:
                    st.error(msg)
        st.stop()

    else:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, user_id, username = login_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.session_state.username = username
                new_session_id = create_new_session(user_id)
                st.session_state.current_session_id = new_session_id
                st.session_state.messages = [
                    {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
                ]
                st.session_state.show_suggestions = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.stop()

# --- Initialize LLM
if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()

# --- Load conversation messages for current session
if "current_session_id" in st.session_state:
    db_messages = get_session_messages(st.session_state.current_session_id)
    if db_messages:
        st.session_state.messages = db_messages
        st.session_state.show_suggestions = False
    elif "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
        ]
        st.session_state.show_suggestions = True
else:
    if "user_id" in st.session_state:
        st.session_state.current_session_id = create_new_session(st.session_state.user_id)
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
        ]
        st.session_state.show_suggestions = True
    else:
        st.session_state.logged_in = False
        st.rerun()

# --- Sidebar with username and sessions
with st.sidebar:
    st.title("💬 Mindful AI")
    if "username" in st.session_state:
        st.markdown(f"### 👤 Welcome, **{st.session_state.username}**!")
    st.write("Your conversations")

    if st.button("➕ New Chat"):
        new_session_id = create_new_session(st.session_state.user_id)
        st.session_state.current_session_id = new_session_id
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
        ]
        st.session_state.show_suggestions = True
        st.rerun()

    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, session_name FROM chat_sessions WHERE user_id=? ORDER BY created_at DESC", (st.session_state.user_id,))
    sessions = cursor.fetchall()
    conn.close()

    st.markdown("<div style='max-height:300px;overflow-y:auto;padding-right:8px;'>", unsafe_allow_html=True)
    for s_id, s_name in sessions:
        msgs = get_session_messages(s_id)
        if len(msgs) == 0:
            continue
        display_name = s_name if s_name else "New Chat"
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(display_name, key=s_id):
                st.session_state.current_session_id = s_id
                st.session_state.messages = msgs
                st.session_state.show_suggestions = False
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del-{s_id}"):
                conn = sqlite3.connect('chat_history.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_messages WHERE session_id=?", (s_id,))
                cursor.execute("DELETE FROM chat_sessions WHERE session_id=?", (s_id,))
                conn.commit()
                conn.close()
                if s_id == st.session_state.current_session_id:
                    new_session_id = create_new_session(st.session_state.user_id)
                    st.session_state.current_session_id = new_session_id
                    st.session_state.messages = [
                        {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
                    ]
                    st.session_state.show_suggestions = True
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.logged_in = False
        st.rerun()

# --- Display chat messages
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><div class="markdown">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><div class="markdown">{message["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Dynamic suggestions after each bot response
if st.session_state.show_suggestions:
    dynamic_suggestions = get_dynamic_suggestions(st.session_state.llm, st.session_state.messages)

    if dynamic_suggestions:
        st.markdown("### 💡 You could also ask:")
        cols = st.columns(len(dynamic_suggestions))
        for idx, suggestion in enumerate(dynamic_suggestions):
            with cols[idx]:
                if st.button(suggestion, key=f"dyn_suggestion_{idx}"):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    save_message(st.session_state.current_session_id, "user", suggestion)
                    st.session_state.show_suggestions = False
                    st.experimental_rerun()

# --- User input handling
prompt = st.chat_input("Type your message here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.current_session_id, "user", prompt)

    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT session_name FROM chat_sessions WHERE session_id=?", (st.session_state.current_session_id,))
    result = cursor.fetchone()
    conn.close()

    current_name = result[0] if result and result[0] else "New Chat"

    if current_name == "New Chat":
        update_session_name_with_llm(st.session_state.current_session_id, prompt, st.session_state.llm)

    chat_history = get_chat_history_for_llm(st.session_state.messages[:-1])
    formatted = prompt_template.invoke({
        "chat_history": chat_history,
        "input": prompt
    })
    assistant_reply = st.session_state.llm.invoke(formatted)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply.content})
    save_message(st.session_state.current_session_id, "assistant", assistant_reply.content)

    st.session_state.show_suggestions = True

    st.experimental_rerun()

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
.chat-messages {
    max-height: calc(100vh - 15rem);
    overflow-y: auto;
    padding: 1rem 2rem;
}

.user-message {
    display: flex;
    justify-content: flex-end;
    margin-left: 2rem;
    margin-bottom: 1rem;
}

.user-message .markdown {
    background: #10a37f;
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 20px 20px 5px 20px;
    max-width: 70%;
    border: none;
    box-shadow: none;
}

.assistant-message {
    display: flex;
    justify-content: flex-start;
    margin-right: 2rem;
    margin-bottom: 1rem;
}

.assistant-message .markdown {
    background: #f5f5f5;
    color: #222;
    padding: 0.8rem 1.2rem;
    border-radius: 20px 20px 20px 5px;
    max-width: 70%;
    border: none;
    box-shadow: none;
}

.stButton button {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

.stButton button:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

.stButton button:hover {
    border: none !important;
    box-shadow: none !important;
}

.stButton button:active {
    border: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)
