import uuid
import sqlite3
from datetime import datetime
import os
import re
import bcrypt
import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# ================== EMAIL VALIDATION ==================
def is_valid_email(email):
    """Validate email format using regex"""
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
        return True, user[0], user[1]  # Return True, user ID, and username
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
        groq_api_key=os.getenv("GROQ_API_KEY", "your_api_key_here"),
        model_name="llama-3.1-8b-instant"
    )

# ================== AUTO SESSION NAMING ==================
def generate_session_name(llm, first_message: str) -> str:
    try:
        instruction = f"Summarize this message into 3-4 words for a chat title:\n\n{first_message}"
        response = llm.invoke(instruction)
        title = response.content.strip()
        return f"Chat about {title}"
    except Exception as e:
        return "New Chat"

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

# ===== LOGIN / REGISTER =====
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

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
    
    else:  # Login
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            success, user_id, username = login_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.session_state.current_session_id = create_new_session(user_id)
                st.session_state.messages = [
                    {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
                ]
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
    ]

if "current_session_id" not in st.session_state:
    if "user_id" in st.session_state:
        st.session_state.current_session_id = create_new_session(st.session_state.user_id)
    else:
        st.session_state.logged_in = False
        st.rerun()

if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("💬 Mindful AI")
    
    # Display username
    if "username" in st.session_state:
        st.markdown(f"### 👤 Welcome, **{st.session_state.username}**!")
    
    st.write("Your conversations")

    if st.button("➕ New Chat"):
        st.session_state.current_session_id = create_new_session(st.session_state.user_id)
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
        ]
        st.rerun()

    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, session_name FROM chat_sessions WHERE user_id=? ORDER BY created_at DESC", (st.session_state.user_id,))
    sessions = cursor.fetchall()
    conn.close()

    st.markdown("<div style='max-height:300px;overflow-y:auto;padding-right:8px;'>", unsafe_allow_html=True)

    for s_id, s_name in sessions:
        msgs = get_session_messages(s_id)
        
        # Skip empty sessions
        if len(msgs) == 0:
            continue
        
        display_name = s_name if s_name else "New Chat"

        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(display_name, key=s_id):
                st.session_state.current_session_id = s_id
                st.session_state.messages = msgs
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
                    st.session_state.current_session_id = create_new_session(st.session_state.user_id)
                    st.session_state.messages = [
                        {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
                    ]
                
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.logged_in = False
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

    formatted = prompt_template.format_messages(input=prompt)
    assistant_reply = st.session_state.llm.invoke(formatted)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply.content})
    save_message(st.session_state.current_session_id, "assistant", assistant_reply.content)

    st.rerun()

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
/* Remove borders and styling from chat messages */
.chat-messages {
    max-height: calc(100vh - 10rem);
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

/* Remove borders from sidebar buttons */
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
