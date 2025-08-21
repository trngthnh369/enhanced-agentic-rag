# enhanced_client.py - Enhanced Streamlit UI with RAG insights
import streamlit as st
import requests
import json
import uuid
import time
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page settings
st.set_page_config(
    page_title="Enhanced Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .message-metadata {
        font-size: 0.8em;
        opacity: 0.8;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    
    .enhanced-badge {
        background: #00d4aa;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.7em;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    .metrics-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set API endpoint
API_ENDPOINT = "http://localhost:5001/chat"
HEALTH_ENDPOINT = "http://localhost:5001/health"
STATS_ENDPOINT = "http://localhost:5001/stats"

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Enhanced Agentic RAG System 2.0</h1>
    <p>Multi-step reasoning • Quality evaluation • Internet search integration</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with system information
with st.sidebar:
    st.title("🔧 System Control")
    
    # System health check
    with st.expander("🏥 System Health", expanded=False):
        if st.button("Check Health"):
            try:
                health_response = requests.get(HEALTH_ENDPOINT, timeout=10)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    st.success("System Healthy ✅")
                    st.json(health_data)
                else:
                    st.error("System Issues ❌")
            except Exception as e:
                st.error(f"Connection Error: {e}")
    
    # System statistics
    with st.expander("📊 System Stats", expanded=False):
        if st.button("Get Statistics"):
            try:
                stats_response = requests.get(STATS_ENDPOINT, timeout=10)
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    
                    # Display key metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Active Threads", 
                                stats_data.get("conversation_stats", {}).get("active_threads", 0))
                        st.metric("Avg Processing Time", 
                                f"{stats_data.get('conversation_stats', {}).get('average_processing_time', 0)}s")
                    
                    with col2:
                        st.metric("Total Messages", 
                                stats_data.get("conversation_stats", {}).get("total_messages", 0))
                    
                    # Agent usage chart
                    agent_usage = stats_data.get("agent_usage", {})
                    if agent_usage:
                        fig = go.Figure(data=[
                            go.Bar(x=list(agent_usage.keys()), y=list(agent_usage.values()))
                        ])
                        fig.update_layout(title="Agent Usage", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Failed to get statistics")
            except Exception as e:
                st.error(f"Stats Error: {e}")
    
    # Conversation controls
    st.markdown("---")
    st.subheader("💬 Conversation")
    
    if st.button("🔄 New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.processing_stats = []
        st.rerun()
    
    # Current thread info
    st.text(f"Thread: {st.session_state.thread_id[:8]}...")
    st.text(f"Messages: {len(st.session_state.messages)}")
    
    # Enhanced features info
    st.markdown("---")
    st.subheader("✨ Enhanced Features")
    st.markdown("""
    - 🧠 Multi-step reasoning
    - 🔄 Query rewriting
    - 📊 Quality evaluation
    - 🌐 Internet search
    - 🏪 Real-time shop data
    - 🎯 Intelligent routing
    """)

# Main chat interface
def display_messages():
    """Display chat messages with enhanced metadata"""
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        
        # Create message container
        with st.container():
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div><strong>👤 You:</strong></div>
                    <div>{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Enhanced assistant message with metadata
                enhanced_badge = '<span class="enhanced-badge">ENHANCED</span>' if message.get("enhanced_processing") else ""
                
                metadata_html = ""
                if "routing_info" in message:
                    routing = message["routing_info"]
                    processing = message.get("processing_info", {})
                    
                    metadata_html = f"""
                    <div class="message-metadata">
                        <strong>🎯 Agent:</strong> {routing.get('agent_used', 'Unknown')}<br>
                        <strong>🎯 Confidence:</strong> {routing.get('confidence', 'Unknown')}<br>
                        <strong>⏱️ Processing Time:</strong> {processing.get('processing_time', 0)}s<br>
                        <strong>🧠 Reasoning:</strong> {routing.get('reasoning', 'N/A')[:100]}...
                    </div>
                    """
                
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div><strong>🤖 Assistant</strong>{enhanced_badge}</div>
                    <div>{content}</div>
                    {metadata_html}
                </div>
                """, unsafe_allow_html=True)

# Display chat history
display_messages()

# Processing metrics visualization
if st.session_state.processing_stats:
    st.markdown("---")
    st.subheader("📈 Processing Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average processing time
        avg_time = sum(stat.get("processing_time", 0) for stat in st.session_state.processing_stats) / len(st.session_state.processing_stats)
        st.metric("Avg Processing Time", f"{avg_time:.2f}s")
    
    with col2:
        # Confidence distribution
        confidences = [stat.get("confidence", "UNKNOWN") for stat in st.session_state.processing_stats]
        high_conf = confidences.count("HIGH")
        st.metric("High Confidence", f"{high_conf}/{len(confidences)}")
    
    with col3:
        # Agent usage
        agents = [stat.get("agent_used", "Unknown") for stat in st.session_state.processing_stats]
        unique_agents = len(set(agents))
        st.metric("Agents Used", unique_agents)
    
    # Processing time chart
    if len(st.session_state.processing_stats) > 1:
        times = [stat.get("processing_time", 0) for stat in st.session_state.processing_stats]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=times, mode='lines+markers', name='Processing Time'))
        fig.update_layout(title="Processing Time Trend", yaxis_title="Seconds", height=300)
        st.plotly_chart(fig, use_container_width=True)

# Chat input
def send_message():
    """Send message with enhanced processing"""
    if st.session_state.user_input:
        user_message = st.session_state.user_input
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user", 
            "content": user_message,
            "timestamp": time.time()
        })
        
        # Clear input
        st.session_state.user_input = ""
        
        # Show processing indicator
        with st.spinner("🧠 Enhanced processing in progress..."):
            # Prepare data for API call
            data = {
                "message": user_message,
                "thread_id": st.session_state.thread_id
            }
            
            try:
                start_time = time.time()
                response = requests.post(API_ENDPOINT, json=data, timeout=60)
                end_time = time.time()
                
                if response.status_code == 200:
                    assistant_response = response.json()
                    
                    # Add assistant response to chat
                    message_data = {
                        "role": "assistant",
                        "content": assistant_response["content"],
                        "timestamp": time.time(),
                        "enhanced_processing": assistant_response.get("enhanced_processing", False),
                        "routing_info": assistant_response.get("routing_info", {}),
                        "processing_info": assistant_response.get("processing_info", {}),
                        "client_processing_time": end_time - start_time
                    }
                    
                    st.session_state.messages.append(message_data)
                    
                    # Add to processing stats
                    stat_data = {
                        "processing_time": assistant_response.get("processing_info", {}).get("processing_time", 0),
                        "confidence": assistant_response.get("routing_info", {}).get("confidence", "UNKNOWN"),
                        "agent_used": assistant_response.get("routing_info", {}).get("agent_used", "Unknown"),
                        "complexity": assistant_response.get("routing_info", {}).get("complexity", "UNKNOWN"),
                        "client_time": end_time - start_time
                    }
                    st.session_state.processing_stats.append(stat_data)
                    
                    # Success feedback
                    st.success(f"✅ Processed by {stat_data['agent_used']} in {stat_data['processing_time']}s")
                    
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏰ Request timeout - Enhanced processing is taking longer than expected")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
        
        st.rerun()

# Before the Quick examples section, add this callback function
def set_example_query(query: str):
    st.session_state.example_query = query

# Quick examples
st.markdown("---")
st.subheader("💡 Try These Enhanced Examples")

example_queries = [
    "Nokia 3210 4G có giá bao nhiêu và có những màu nào?",
    "So sánh Samsung Galaxy A05s với Nokia 3210 về camera và pin",
    "Cửa hàng có những chi nhánh nào và giờ mở cửa như thế nào?",
    "Tôi muốn mua điện thoại dưới 2 triệu, có gợi ý nào không?",
    "Chính sách bảo hành và đổi trả của cửa hàng ra sao?"
]

# Initialize example_query in session state if not present
if 'example_query' not in st.session_state:
    st.session_state.example_query = ''

cols = st.columns(2)
for i, query in enumerate(example_queries):
    with cols[i % 2]:
        if st.button(f"📱 {query}", key=f"example_{i}", on_click=set_example_query, args=(query,)):
            pass

# Update the chat input form to use the example query
st.markdown("---")
with st.form(key="enhanced_chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Use the example query as the default value if available
        st.text_input(
            "Ask about products, shop info, or anything else:",
            key="user_input",
            placeholder="Example: Nokia 3210 4G có giá bao nhiêu? Cửa hàng mở cửa lúc nào?",
            help="Enhanced RAG will analyze your query and choose the best approach",
            value=st.session_state.example_query
        )
        # Clear the example query after it's used
        st.session_state.example_query = ''
    
    with col2:
        submit_button = st.form_submit_button(
            "🚀 Send", 
            on_click=send_message,
            use_container_width=True
        )

# Footer with system info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    🤖 Enhanced Agentic RAG System 2.0 | Powered by Google Gemini 2.0 Flash<br>
    ✨ Features: Multi-step reasoning, Quality evaluation, Internet search
</div>
""", unsafe_allow_html=True)