import re
import time
import psycopg_pool
import streamlit as st
from src.config import DB_URI
from src.agent import create_graph
from src.engine import get_vector_store
from src.processor import run_ingestion_pipeline
from src.keep_alive import start_keep_alive_service
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image

# Security: Disable the vulnerable PSD decoder entirely at the app level
# This prevents potential image-based attacks by blocking PSD file processing
if hasattr(Image, "register_extension"):
    Image.register_extension('PSD', '')
    # This prevents Pillow from even attempting to open a .psd file

# --- 1. Page Configuration & Professional Branding ---
# Configure the Streamlit app with legal-themed styling and responsive layout
st.set_page_config(
    page_title="LegisYukti — An Agentic RAG Framework for Multi-Document Reasoning",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Health check endpoint — must be checked immediately after page config so the
# keep-alive self-ping returns quickly without triggering heavy UI rendering.
if st.query_params.get("health") == "true":
    st.json({"status": "healthy", "timestamp": time.time(), "service": "legisyukti"})
    st.stop()

# Auto-scroll JavaScript injection for smooth conversation flow
# Monitors DOM changes and automatically scrolls to show latest messages
st.components.v1.html("""<script>var body = window.parent.document.querySelector(".main");new MutationObserver(function() {body.scrollTop = body.scrollHeight;}).observe(body, {attributes: true, childList: true, subtree: true});</script>""", height=0)

# Professional UI styling with dark theme and legal color scheme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    .stApp { background: #0f172a; color: #f1f5f9; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }

    /* Interactive UI elements with hover effects and legal color scheme */
    .stButton>button { border-radius: 8px; font-weight: 600; letter-spacing: 0.5px; transition: all 0.3s ease; border: 1px solid rgba(56, 189, 248, 0.2); width: 100%; }
    .stButton>button:hover { border-color: #38bdf8; color: #38bdf8; background: rgba(56, 189, 248, 0.05); }

    /* Clean chat message styling without horizontal dividers for better readability */
    .stChatMessage { border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 25px !important; margin-bottom: 25px !important; }

    /* Quality assessment cards with color-coded fidelity indicators */
    .fidelity-card { margin-top: 15px; padding: 20px; border-radius: 12px; font-size: 13.5px; line-height: 1.6; border-left: 10px solid; background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.05); }
    .fid-high { border-left-color: #10b981; } .fid-med { border-left-color: #f59e0b; } .fid-low { border-left-color: #ef4444; }
    .score-pill { background: #fff; color: #020617; padding: 3px 10px; border-radius: 6px; font-weight: 800; margin-right: 12px; font-family: monospace; }
    .fidelity-status { text-transform: uppercase; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-bottom: 5px; display: block; }
    .status-high { color: #10b981; } .status-med { color: #f59e0b; } .status-low { color: #ef4444; }

    .sidebar-label { color: #38bdf8; font-size: 11px; font-weight: 800; text-transform: uppercase; margin-top: 30px; margin-bottom: 12px; letter-spacing: 1.5px; }
    .engine-card { background: linear-gradient(145deg, #0f172a, #1e293b); padding: 15px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3); margin-bottom: 15px; }
    .pulse-dot { height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 8px #10b981; }
    .knowledge-disclosure { font-size: 10.5px; color: #94a3b8; margin-top: 12px; line-height: 1.5; font-style: italic; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px; }

    .disclaimer-hero { border: 1px solid #ef4444; padding: 16px; border-radius: 10px; color: #fca5a5; font-size: 12px; background: rgba(239, 68, 68, 0.05); margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Session State Initialization ---
# Initialize persistent session state for workspace management and conversation tracking
if "workspace" not in st.session_state:
    st.session_state.workspace = st.query_params.get("checkpoint", "DEFAULT")  # Current active workspace/case
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {"messages": []}  # Cached conversation state
if "busy" not in st.session_state:
    st.session_state.busy = False  # Processing state flag

# --- 3. System Core Initialization (Cached) ---
# Initialize the core legisyukti app with database connections and LangGraph agent
@st.cache_resource
def init_system_core():
    # Establish PostgreSQL connection pool for conversation persistence
    pool = psycopg_pool.ConnectionPool(
        conninfo=DB_URI, max_size=10, min_size=2,
        check=psycopg_pool.ConnectionPool.check_connection,
        kwargs={"sslmode": "require", "connect_timeout": 10}
    )

    # Initialize database schema for workspace and audit logging
    with pool.connection() as conn:
        conn.autocommit = True
        PostgresSaver(conn).setup()  # Create LangGraph checkpoint tables
        with conn.cursor() as cur:
            # Create workspace registry for case management
            cur.execute("CREATE TABLE IF NOT EXISTS workspace_registry (thread_id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
            cur.execute("INSERT INTO workspace_registry (thread_id) VALUES ('DEFAULT') ON CONFLICT DO NOTHING;")
            # Create audit logs table for response quality tracking
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='audit_logs' AND column_name='message_id';")
            if not cur.fetchone():
                cur.execute("DROP TABLE IF EXISTS audit_logs;")
                cur.execute("CREATE TABLE audit_logs (thread_id TEXT, message_id TEXT, score INTEGER, feedback TEXT, PRIMARY KEY (thread_id, message_id));")

    # Initialize vector store for legal document retrieval
    vector_store = get_vector_store()
    #ingestion check - if no points, trigger ingestion pipeline to populate the knowledge base
    try:
        # Check point count in your Qdrant collection
        status = vector_store.client.get_collection("indian_legal_library")
        if status.points_count == 0:
            with st.spinner("📥 Legal Knowledge Base is empty. Re-ingesting 17 Statutes..."):
                run_ingestion_pipeline(vector_store)
    except Exception:
        # If collection doesn't even exist, trigger pipeline
        with st.spinner("🆕 Initializing Statutory Library for the first time..."):
            run_ingestion_pipeline(vector_store)
    # Create and return the LangGraph agent with PostgreSQL checkpointing
    return pool, create_graph(PostgresSaver(pool))

# Initialize the core system components
pool, graph = init_system_core()

# This prevents the app and Qdrant Cloud from sleeping due to inactivity
keep_alive_service = start_keep_alive_service(interval_minutes=10)

# --- 4. Persistent Logic Utilities ---
# Utility functions for workspace management and audit logging

def normalize_id(text):
    """Normalize workspace IDs by removing special characters and converting to uppercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).upper()

def fetch_all_workspaces():
    """Retrieve all available workspaces from database registry and active checkpoints."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Union of registered workspaces and active conversation threads
                cur.execute("SELECT DISTINCT thread_id FROM workspace_registry UNION SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC;")
                return [r[0] for r in cur.fetchall()]
    except: return ["DEFAULT"]

def get_audit_map(thread_id):
    """Retrieve quality audit scores and feedback for all messages in a workspace."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT message_id, score, feedback FROM audit_logs WHERE thread_id = %s", (thread_id,))
                return {r[0]: {"score": r[1], "feedback": r[2]} for r in cur.fetchall()}
    except: return {}

# --- 5. Execution Engine (Streaming + Immediate Audit Sync) ---
# Main function to process legal queries through the LangGraph agent pipeline

def run_research_logic(prompt, is_regen=False):
    """
    Execute the complete legal research pipeline with streaming responses.

    Args:
        prompt: User's legal query
        is_regen: Whether this is a regeneration of a previous response
    """
    st.session_state.busy = True  # Set processing flag

    # Configure thread ID for conversation persistence
    config = {"configurable": {"thread_id": st.session_state.workspace}}

    # Handle regeneration by removing the last AI response
    if is_regen:
        state = graph.get_state(config)
        if state.values and "messages" in state.values:
            msgs = state.values["messages"]
            if isinstance(msgs[-1], AIMessage):
                graph.update_state(config, {"messages": msgs[:-1]})

    # Execute the LangGraph agent with streaming message output
    status_ui = st.status("📡 Engaging Statutory Intelligence Core...", expanded=False)
    advice_buffer = ""

    # Stream processing with message filtering (exclude internal tool messages)
    for chunk, metadata in graph.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="messages"):
        node = metadata.get("langgraph_node", "")
        if isinstance(chunk, AIMessage) and node in ["generate_response", "final_answer"]:
            # Filter out internal tool messages (context storage notifications)
            if any(x in chunk.content.lower() for x in ["context stored", "penalty:"]): continue
            advice_buffer = chunk.content

    status_ui.update(label="✅ Analysis Complete", state="complete")

    # Display the response in chat format
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        # Simulate streaming effect for better UX
        for word in advice_buffer.split(' '):
            streamed += word + " "
            placeholder.markdown(streamed + "▌")  # Cursor effect
            time.sleep(0.01)

        # Immediately persist audit data for UI display
        final_state = graph.get_state(config).values
        final_msgs = final_state.get("messages", [])
        if final_msgs and isinstance(final_msgs[-1], AIMessage):
            msg_id = final_msgs[-1].id
            sc = final_state.get("evaluation_score", 0)
            fb = final_state.get("evaluation_feedback", "Audited successfully.")
            # Store audit data in database for persistence across sessions
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO audit_logs (thread_id, message_id, score, feedback) 
                                   VALUES (%s, %s, %s, %s) ON CONFLICT (thread_id, message_id) 
                                   DO UPDATE SET score=EXCLUDED.score, feedback=EXCLUDED.feedback;""",
                                (st.session_state.workspace, msg_id, sc, fb))

    st.session_state.graph_state = final_state
    st.session_state.busy = False
    st.rerun()  # Refresh UI to show updated state

# --- 6. Sidebar Implementation (Workspace Management) ---
# Professional sidebar with workspace management, legal knowledge base display, and operational guidance

with st.sidebar:
    st.markdown('<h1 style="color:#38bdf8; text-align:center; font-size:24px;">Workspace Management</h1>', unsafe_allow_html=True)
    if st.button("➕ REGISTER NEW CASE", type="primary"):
        @st.dialog("Register Workspace")
        def reg_dialog():
            new_id = st.text_input("New ID", placeholder="CASE-2024").strip()
            if st.button("Initialize"):
                if not new_id: st.error("ID required."); return
                all_ws = fetch_all_workspaces()
                if any(normalize_id(ws) == normalize_id(new_id) for ws in all_ws):
                    st.warning("⚠️ Workspace already exists.")
                else:
                    with pool.connection() as conn:
                        with conn.cursor() as cur: cur.execute("INSERT INTO workspace_registry (thread_id) VALUES (%s);", (new_id.upper(),))
                    st.query_params["checkpoint"] = new_id.upper()
                    if "graph_state" in st.session_state: del st.session_state["graph_state"]
                    st.rerun()
        reg_dialog()
    
    st.markdown('<div class="sidebar-label">Case Navigator</div>', unsafe_allow_html=True)
    all_ws = fetch_all_workspaces(); cq = st.query_params.get("checkpoint", "DEFAULT")
    if st.session_state.workspace != cq:
        if "graph_state" in st.session_state: del st.session_state["graph_state"]
    st.session_state.workspace = cq
    if st.session_state.workspace not in all_ws: all_ws.insert(0, st.session_state.workspace)
    
    idx = all_ws.index(st.session_state.workspace)
    selected_ws = st.selectbox("Active Case", all_ws, index=idx, label_visibility="collapsed")
    if selected_ws != st.session_state.workspace:
        st.query_params["checkpoint"] = selected_ws
        if "graph_state" in st.session_state: del st.session_state["graph_state"]
        st.rerun()

    st.markdown('<div class="sidebar-label">Engine Analytics</div>', unsafe_allow_html=True)
    st.markdown(f'''<div class="engine-card"><div style="font-size:12px; margin-bottom:10px;"><span class="pulse-dot"></span><b>Vector Store:</b> Qdrant SSL ✅</div><div style="font-size:11px; color:#94a3b8;"><b>Index Depth:</b> 17 core PDFs<br><b>Status:</b> Ready</div></div>''', unsafe_allow_html=True)

    # Keep-Alive Service Status for Hugging Face Spaces
    from src.keep_alive import get_keep_alive_status
    keep_alive_status = get_keep_alive_status()
    if keep_alive_status:
        status_icon = "🟢" if keep_alive_status["running"] else "🔴"
        time_since = keep_alive_status.get("time_since_last_ping")
        time_str = f"{time_since:.1f}min ago" if time_since else "Never"
        st.markdown(f'''<div class="engine-card"><div style="font-size:12px; margin-bottom:10px;"><span style="color:#10b981;">{status_icon}</span><b> Keep-Alive Service:</b> Active</div><div style="font-size:11px; color:#94a3b8;"><b>Interval:</b> {keep_alive_status["interval_seconds"]//60}min<br><b>Last Ping:</b> {time_str}</div></div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="engine-card"><div style="font-size:12px; margin-bottom:10px;"><span style="color:#ef4444;">🔴</span><b> Keep-Alive Service:</b> Inactive</div><div style="font-size:11px; color:#94a3b8;">Free tier protection disabled</div></div>''', unsafe_allow_html=True)
    
    with st.expander("📚 LEGAL KNOWLEDGE BASE"):
        st.markdown("""
        * **Bharatiya Nyaya Sanhita 2023**
        * **Bharatiya Nagarik Suraksha Sanhita 2023**
        * **Bharatiya Sakshya Adhiniyam 2023**
        * **Code of Civil Procedure 1908**
        * **Constitution of India 1950**
        * **Indian Contract Act 1872**
        * **Transfer of Property Act 1882**
        * **Code on Wages 2019**
        * **Consumer Protection Act 2019**
        * **IT Act 2000**
        * **Negotiable Instruments Act 1881**
        * **Special Marriage Act 1954**
        * **NDPS Act 1985**
        * **POCSO Act 2012**
        * **Hindu Marriage Act 1955**
        * **Indian Succession Act 1925**
        * **Registration Act 1908**
        
        <div class="knowledge-disclosure">
        <b>RAG Advisory:</b> Intelligence is grounded in semantic indexing of these 17 official frameworks to ensure statutory alignment.
        </div>""", unsafe_allow_html=True)

    with st.expander("📘 OPERATIONAL GUIDE"):
        st.markdown("""<div style="font-size:11px;">
        1. <b>Workspace Strategy:</b> Use 1 workspace for 1 type of advisory (e.g. Criminal vs Civil) for maximum semantic focus.<br><br>
        2. <b>Audit Check:</b> Review the Fidelity Card at the bottom of each block. If low, click 'Regenerate Analysis'.<br><br>
        <b>Prompt Examples:</b><br>
        - <i>'Define 'Community Service' as a new form of punishment in the BNSS'</i><br>
        - <i>'What are the safe harbor protections for 'Intermediaries' under the IT Act?'</i><br>
        - <i>'How is 'Digital Evidence' authenticated under Section 63 of the BSA?'</i>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button(f"Purge {st.session_state.workspace}"):
        @st.dialog("Purge Case Data")
        def purge_dialog():
            st.error(f"⚠️ PERMANENT ACTION: Wipe ALL records for: **{st.session_state.workspace}**?")
            if st.button("🔥 Confirm Purge", type="primary", use_container_width=True):
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        for tbl in ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "audit_logs", "workspace_registry"]:
                            cur.execute(f"DELETE FROM {tbl} WHERE thread_id = %s", (st.session_state.workspace,))
                        cur.execute("INSERT INTO workspace_registry (thread_id) VALUES ('DEFAULT') ON CONFLICT DO NOTHING;")
                st.query_params["checkpoint"] = "DEFAULT"
                st.session_state.graph_state = {"messages": []}
                st.rerun()
        purge_dialog()

# --- 7. Main Chat Interface Rendering ---
# Render the conversation history with audit scores and regeneration capabilities

st.title("⚖️ LegisYukti — An Agentic RAG Framework for Multi-Document Reasoning")
st.markdown('<div class="disclaimer-hero"><b>⚠️ MANDATORY DISCLOSURE:</b> AI research or reasoning tool only. Statutory results must be cross-verified with Official Gazettes. No attorney-client relationship is formed.</div>', unsafe_allow_html=True)

# Only render when not processing a query to prevent UI conflicts
if not st.session_state.busy:
    config_render = {"configurable": {"thread_id": st.session_state.workspace}}

    # Load conversation state from database if not cached
    if "graph_state" not in st.session_state or not st.session_state.graph_state.get("messages"):
        try:
            st.session_state.graph_state = graph.get_state(config_render).values
        except:
            st.session_state.graph_state = {"messages": []}

    raw = st.session_state.graph_state.get("messages", [])

    # Process message history: group consecutive AI messages and filter internal messages
    history, ai_buf = [], []
    for m in raw:
        if isinstance(m, HumanMessage):
            if ai_buf:
                history.append(ai_buf[-1])  # Save the last AI message before user message
            ai_buf, history = [], history + [m]  # Reset buffer and add user message
        elif isinstance(m, AIMessage):
            # Filter out internal system messages (penalty calculations, verdicts)
            if not any(x in m.content.lower() for x in ["penalty:", "verdict:"]):
                ai_buf.append(m)  # Buffer AI messages
    if ai_buf:
        history.append(ai_buf[-1])  # Add final buffered AI message

    # Display appropriate UI based on conversation state
    if not history:
        st.info(f"Workspace **{st.session_state.workspace}** active. Submit a query to begin.")
    else:
        # Load audit scores for quality assessment display
        audit_history = get_audit_map(st.session_state.workspace)
        for i, msg in enumerate(history):
            with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
                st.markdown(msg.content)

                # Display quality audit card for AI responses
                if isinstance(msg, AIMessage):
                    msg_id = getattr(msg, 'id', None)
                    if msg_id and msg_id in audit_history:
                        audit = audit_history[msg_id]
                        s, f = audit["score"], audit["feedback"]

                        # Determine quality status and recommendations
                        if s >= 8:
                            status_lbl, cls, info = "HIGH CONFIDENCE", "fid-high", "Direct statutory match."
                        elif s >= 5:
                            status_lbl, cls, info = "MODERATE MATCH", "fid-med", "Verify specific sub-sections."
                        else:
                            status_lbl, cls, info = "ACTION: TRIGGER REGENERATE", "fid-low", "Low fidelity detected. Please click the button below."

                        # Render quality assessment card with color coding
                        st.markdown(f'''<div class="fidelity-card {cls}">
                            <span class="fidelity-status status-{cls.split("-")[1]}">{status_lbl}</span>
                            <span class="score-pill">SCORE: {s}/10</span>
                            <div style="margin-top:8px;"><b>Instruction:</b> {info}</div>
                            <div style="margin-top:4px; font-size:12px; color:#94a3b8;"><b>Audit Feedback:</b> {f}</div>
                        </div>''', unsafe_allow_html=True)

                    # Regeneration button for low-quality responses
                    if st.button("🔄 REGENERATE ANALYSIS", key=f"re_{i}"):
                        # Find the corresponding user query for regeneration
                        st.session_state.regen_prompt = next((m.content for m in reversed(history[:i+1]) if isinstance(m, HumanMessage)), None)
                        st.rerun()

# Handle regeneration requests
if "regen_prompt" in st.session_state:
    prompt = st.session_state.pop("regen_prompt")
    run_research_logic(prompt, is_regen=True)

# Main chat input for new queries
if query := st.chat_input(f"Query {st.session_state.workspace}..."):
    with st.chat_message("user"):
        st.markdown(query)
    run_research_logic(query)