import uuid
import os
import re
import bcrypt
import streamlit as st
from datetime import datetime
from pymongo import MongoClient

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ================== EMAIL VALIDATION ==================
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[a-zA-Z\d-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# ================== DATABASE CONNECTION WITH CACHING ==================
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection once and cache it"""
    mongo_uri = st.secrets["MONGO_URI"]
    client = MongoClient(mongo_uri)
    return client["chat_app"]

def get_db():
    """Get cached database connection"""
    return init_mongodb()

# ================== USERS ==================
def register_user(username, email, password):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    db = get_db()
    users = db["users"]

    if users.find_one({"email": email}):
        return False, "Email already exists"
    
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "username": username,
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.utcnow()
    })
    return True, "Registration successful"

def login_user(email, password):
    db = get_db()
    users = db["users"]
    user = users.find_one({"email": email})
    
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return True, str(user["_id"]), user["username"]
    return False, None, None

# ================== CHAT SESSIONS ==================
def create_new_session(user_id, session_name="New Chat"):
    db = get_db()
    sessions = db["chat_sessions"]
    
    session_id = str(uuid.uuid4())
    sessions.insert_one({
        "user_id": user_id,
        "session_id": session_id,
        "session_name": session_name,
        "created_at": datetime.utcnow()
    })
    return session_id

def update_session_name_with_llm(session_id, first_message, llm):
    db = get_db()
    sessions = db["chat_sessions"]
    session_name = generate_session_name(llm, first_message)
    sessions.update_one({"session_id": session_id}, {"$set": {"session_name": session_name}})

# ================== MESSAGES ==================
def save_message(session_id, message_type, content):
    db = get_db()
    messages = db["chat_messages"]
    message_text = getattr(content, "content", str(content))
    messages.insert_one({
        "session_id": session_id,
        "message_type": message_type,
        "content": message_text,
        "timestamp": datetime.utcnow()
    })

def get_session_messages(session_id):
    db = get_db()
    messages = db["chat_messages"].find({"session_id": session_id}).sort("timestamp", 1)
    
    results = []
    for m in messages:
        results.append({
            "role": "user" if m["message_type"] == "user" else "assistant",
            "content": m["content"]
        })
    return results

# ================== LLM WITH CONVERSATION MEMORY ==================
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

@st.cache_resource
def initialize_llm():
    """Cache LLM instance"""
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

# ================== AUTO SESSION NAMING (ASYNC) ==================
def generate_session_name(llm, first_message: str) -> str:
    try:
        instruction = f"Summarize this message into 3-4 words for a chat title:\n\n{first_message}"
        response = llm.invoke(instruction)
        title = response.content.strip()
        return f"Chat about {title[:30]}"  # Limit title length
    except Exception:
        return "New Chat"

# ================== HANDLE SUGGESTION CLICK ==================
def handle_suggestion(suggestion_text):
    st.session_state.messages.append({"role": "user", "content": suggestion_text})
    save_message(st.session_state.current_session_id, "user", suggestion_text)

    # Check and update session name only if needed
    db = get_db()
    session = db["chat_sessions"].find_one({"session_id": st.session_state.current_session_id}, {"session_name": 1})
    
    if session and session.get("session_name") == "New Chat":
        update_session_name_with_llm(st.session_state.current_session_id, suggestion_text, st.session_state.llm)

    chat_history = get_chat_history_for_llm(st.session_state.messages[:-1])
    formatted = prompt_template.invoke({
        "chat_history": chat_history,
        "input": suggestion_text
    })
    assistant_reply = st.session_state.llm.invoke(formatted)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply.content})
    save_message(st.session_state.current_session_id, "assistant", assistant_reply.content)

    st.session_state.show_suggestions = True

# ================== STREAMLIT APP ==================
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

# ========== INITIALIZE SESSION STATE ==========
if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

if "current_session_id" in st.session_state:
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        db_messages = get_session_messages(st.session_state.current_session_id)
        if db_messages:
            st.session_state.messages = db_messages
            st.session_state.show_suggestions = True
        else:
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

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("💬 Mindful AI")
    
    if "username" in st.session_state:
        st.markdown(f"### 👤 {st.session_state.username}")
    
    st.write("Your conversations")

    if st.button("➕ New Chat"):
        new_session_id = create_new_session(st.session_state.user_id)
        st.session_state.current_session_id = new_session_id
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm Mindful AI, your compassionate support companion. How can I help you today?"}
        ]
        st.session_state.show_suggestions = True
        st.rerun()

    db = get_db()
    # Limit sessions to last 20 for performance
    sessions = list(db["chat_sessions"].find({"user_id": st.session_state.user_id}).sort("created_at", -1).limit(20))

    st.markdown("<div style='max-height:300px;overflow-y:auto;padding-right:8px;'>", unsafe_allow_html=True)

    for s in sessions:
        # Check if session has messages (avoid extra query)
        msg_count = db["chat_messages"].count_documents({"session_id": s["session_id"]}, limit=1)
        if msg_count == 0:
            continue

        display_name = s.get("session_name", "New Chat")[:40]  # Truncate long names

        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(display_name, key=s["session_id"]):
                st.session_state.current_session_id = s["session_id"]
                st.session_state.messages = get_session_messages(s["session_id"])
                st.session_state.show_suggestions = True
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del-{s['session_id']}"):
                db["chat_messages"].delete_many({"session_id": s["session_id"]})
                db["chat_sessions"].delete_one({"session_id": s["session_id"]})
                if s["session_id"] == st.session_state.current_session_id:
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

# ========== CHAT UI ==========
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ========== SUGGESTION BUTTONS ==========
if st.session_state.show_suggestions and len(st.session_state.messages) <= 1:
    st.markdown("### 💡 Try asking about:")
    
    suggestions = [
        "😰 I'm feeling anxious",
        "😔 Cope with stress?",
        "🧘 Mindfulness tips"
    ]
    
    cols = st.columns(len(suggestions))
    for idx, suggestion in enumerate(suggestions):
        with cols[idx]:
            if st.button(suggestion, key=f"suggestion_{idx}"):
                handle_suggestion(suggestion)
                st.rerun()

# ========== USER INPUT ==========
prompt = st.chat_input("Type your message here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.current_session_id, "user", prompt)

    db = get_db()
    session = db["chat_sessions"].find_one({"session_id": st.session_state.current_session_id}, {"session_name": 1})
    current_name = session.get("session_name", "New Chat") if session else "New Chat"

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
    st.rerun()
