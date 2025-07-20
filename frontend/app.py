"""
Streamlit Frontend voor RAG Contract Analyzer
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import re
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from config import GeminiConfig, VectorStoreConfig
from utils.document_processor import ContractProcessor, validate_pdf_file
from utils.vector_store import VectorStoreManager
from chains.contract_analyzer_chain import ContractAnalyzerChain, ContractConversation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Contract Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/rag-contract-analyzer',
        'Report a bug': 'https://github.com/yourusername/rag-contract-analyzer/issues',
        'About': """
        # RAG Contract Analyzer
        
        Intelligente contractanalyse met AI
        
        **Features:**
        - PDF contract processing
        - Semantic search & Q&A
        - Risk analysis
        - Compliance checking
        - Executive summaries
        
        **Tech Stack:**
        - Gemini API + LangChain
        - ChromaDB vector store
        - Streamlit frontend
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme colors */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Risk level indicators */
    .risk-high {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #e74c3c;
    }
    
    .risk-medium {
        background: linear-gradient(45deg, #feca57, #ff9ff3);
        color: #2c3e50;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #f39c12;
    }
    
    .risk-low {
        background: linear-gradient(45deg, #48dbfb, #0abde3);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #00a8ff;
    }
    
    .success-box {
        background: linear-gradient(45deg, #55a3ff, #003d82);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #27ae60;
    }
    
    /* Source citations */
    .source-citation {
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
        padding: 10px;
        margin: 10px 0;
        font-style: italic;
        font-size: 0.9em;
        border-radius: 0 5px 5px 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid #e1e8ed;
    }
    
    .user-message {
        background-color: #f0f8ff;
        border-left: 4px solid #007bff;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
    }
    
    /* Progress indicators */
    .progress-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Action buttons */
    .action-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        margin: 0.2rem;
    }
    
    /* Compliance indicators */
    .compliance-pass {
        color: #27ae60;
        font-weight: bold;
    }
    
    .compliance-fail {
        color: #e74c3c;
        font-weight: bold;
    }
    
    .compliance-partial {
        color: #f39c12;
        font-weight: bold;
    }
    
    /* Hide streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'vector_store': None,
        'analyzer_chain': None,
        'conversation': None,
        'current_contract': None,
        'chat_history': [],
        'contract_stats': {},
        'processing_status': 'idle',
        'risk_analysis_cache': {},
        'compliance_cache': {},
        'summary_cache': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Utility functions
def format_currency(amount: str) -> str:
    """Format currency amounts"""
    try:
        # Extract numbers from string
        numbers = re.findall(r'[\d,.]+', amount)
        if numbers:
            # Take the largest number (likely the main amount)
            num_str = max(numbers, key=len)
            num_str = num_str.replace(',', '.')
            try:
                num = float(num_str)
                if num >= 1000:
                    return f"€ {num:,.0f}"
                else:
                    return f"€ {num:.2f}"
            except:
                return amount
    except:
        return amount
    return amount

def extract_risk_score(text: str) -> Optional[float]:
    """Extract risk score from analysis text"""
    patterns = [
        r'RISICO SCORE:\s*(\d+(?:\.\d+)?)/10',
        r'Risk Score:\s*(\d+(?:\.\d+)?)/10',
        r'Score:\s*(\d+(?:\.\d+)?)/10'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def create_risk_gauge(score: float) -> go.Figure:
    """Create a risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score"},
        delta = {'reference': 5},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_compliance_chart(compliance_data: Dict) -> go.Figure:
    """Create compliance overview chart"""
    if not compliance_data.get('results'):
        return go.Figure()
    
    results = compliance_data['results']
    
    statuses = []
    rules = []
    colors = []
    
    for result in results:
        status = result.get('compliance_status', 'ONDUIDELIJK')
        rule = result.get('rule_description', 'Unknown')
        
        statuses.append(status)
        rules.append(rule[:30] + '...' if len(rule) > 30 else rule)
        
        if status == 'VOLDOET':
            colors.append('#27ae60')
        elif status == 'VOLDOET NIET':
            colors.append('#e74c3c')
        else:
            colors.append('#f39c12')
    
    fig = go.Figure(data=[
        go.Bar(
            y=rules,
            x=[1] * len(rules),
            orientation='h',
            marker_color=colors,
            text=statuses,
            textposition="middle center",
            textfont=dict(color="white", size=12)
        )
    ])
    
    fig.update_layout(
        title="Compliance Overview",
        xaxis_title="",
        yaxis_title="Rules",
        showlegend=False,
        height=max(300, len(rules) * 50),
        xaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=200, r=50, t=50, b=50)
    )
    
    return fig

def create_contract_timeline(summary_text: str) -> go.Figure:
    """Create timeline from contract summary"""
    # Mock timeline data - in real implementation, extract from summary
    timeline_data = [
        {"event": "Contract getekend", "date": datetime.now().strftime("%Y-%m-%d"), "type": "milestone", "status": "completed"},
        {"event": "Financiering regelen", "date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"), "type": "deadline", "status": "pending"},
        {"event": "Bouwkundige keuring", "date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"), "type": "action", "status": "pending"},
        {"event": "Notaris transport", "date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"), "type": "milestone", "status": "future"},
        {"event": "Oplevering", "date": (datetime.now() + timedelta(days=75)).strftime("%Y-%m-%d"), "type": "milestone", "status": "future"}
    ]
    
    df = pd.DataFrame(timeline_data)
    df['date'] = pd.to_datetime(df['date'])
    
    colors = {'milestone': '#007bff', 'deadline': '#dc3545', 'action': '#ffc107'}
    
    fig = px.timeline(
        df,
        x_start="date",
        x_end="date", 
        y="event",
        color="type",
        color_discrete_map=colors,
        title="Contract Timeline"
    )
    
    fig.update_layout(height=300)
    return fig

# Main app components
def show_header():
    """Show main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Contract Analyzer</h1>
        <p>Intelligente contractanalyse met AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar():
    """Show sidebar with configuration and stats"""
    with st.sidebar:
        st.markdown("## ⚙️ Configuratie")
        
        # Model settings
        model = st.selectbox(
            "🧠 AI Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            help="Gemini-1.5-flash is sneller, Gemini-1.5-pro is accurater"
        )
        
        # Vector store settings
        use_cloud = st.checkbox(
            "☁️ Cloud Vector Store", 
            value=False,
            help="Gebruik Pinecone voor productie (vereist API key)"
        )
        
        # Contract type
        contract_type = st.selectbox(
            "📋 Contract Type",
            ["Koopovereenkomst", "Huurovereenkomst", "ROZ-contract", "Aannemingsovereenkomst"]
        )
        
        # Client role
        client_role = st.radio(
            "👤 Perspectief",
            ["Koper", "Verkoper", "Huurder", "Verhuurder"],
            help="Vanuit welk perspectief analyseren we?"
        )
        
        st.markdown("---")
        
        # System status
        st.markdown("## 📊 Systeem Status")
        
        # API status
        try:
            if GeminiConfig.API_KEY:
                st.success("✅ Gemini API Connected")
            else:
                st.error("❌ Gemini API Not Configured")
        except:
            st.error("❌ Configuration Error")
        
        # Contract stats
        if st.session_state.current_contract:
            contract = st.session_state.current_contract
            st.markdown("### 📄 Current Contract")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Pages", contract.get('pages', 0))
                st.metric("📝 Words", f"{contract.get('words', 0):,}")
            with col2:
                st.metric("🧩 Chunks", contract.get('chunks', 0))
                st.metric("⭐ Quality", f"{contract.get('avg_quality', 0):.2f}")
        
        # Memory usage
        if st.session_state.vector_store:
            try:
                stats = st.session_state.vector_store.get_collection_stats()
                st.markdown("### 🗄️ Vector Store")
                st.metric("📚 Documents", stats.get('total_documents', 0))
                st.metric("🔢 Dimensions", stats.get('embedding_dimension', 768))
            except:
                pass
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        
        if st.button("🗑️ Reset System", help="Clear all data and start fresh"):
            for key in st.session_state.keys():
                if key not in ['vector_store', 'analyzer_chain']:
                    del st.session_state[key]
            st.rerun()
        
        if st.session_state.vector_store and st.button("📊 Export Results"):
            # Create export data
            export_data = {
                "contract": st.session_state.current_contract,
                "chat_history": st.session_state.chat_history,
                "export_time": datetime.now().isoformat()
            }
            
            st.download_button(
                "💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"contract_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Debug info
        if st.checkbox("🐛 Debug Info"):
            st.json({
                "session_keys": list(st.session_state.keys()),
                "vector_store_status": st.session_state.vector_store is not None,
                "analyzer_status": st.session_state.analyzer_chain is not None,
                "chat_messages": len(st.session_state.chat_history)
            })
    
    return {
        "model": model,
        "use_cloud": use_cloud,
        "contract_type": contract_type,
        "client_role": client_role
    }

def upload_tab():
    """File upload and processing tab"""
    st.header("📤 Contract Upload & Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload een contract (PDF)",
            type=['pdf'],
            help="Ondersteunde formaten: PDF (max 50MB)"
        )
        
        if uploaded_file:
            # File info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📄 **{uploaded_file.name}** ({file_size:.1f} MB)")
            
            # Processing options
            st.markdown("### ⚙️ Processing Options")
            
            col1a, col1b = st.columns(2)
            with col1a:
                chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
                contract_type = st.selectbox("Contract Type", 
                    ["koopovereenkomst", "huurovereenkomst", "roz-contract"])
            
            with col1b:
                chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200, 50)
                quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.5, 0.1)
    
    with col2:
        st.markdown("### 📋 Processing Steps")
        processing_steps = [
            "1. 📄 PDF Text Extraction",
            "2. 🧹 Text Cleaning & Normalization", 
            "3. 🔍 Section Identification",
            "4. 🧩 Intelligent Chunking",
            "5. 🏷️ Semantic Tagging",
            "6. 🧮 Embedding Generation",
            "7. 🗄️ Vector Store Upload",
            "8. ✅ System Ready"
        ]
        
        for step in processing_steps:
            st.markdown(f"- {step}")
    
    if uploaded_file and st.button("🚀 Process Contract", type="primary"):
        process_contract(uploaded_file, chunk_size, chunk_overlap, contract_type, quality_threshold)

def process_contract(uploaded_file, chunk_size, chunk_overlap, contract_type, quality_threshold):
    """Process uploaded contract"""
    
    # Save uploaded file
    contract_path = Path("contracts/uploaded") / uploaded_file.name
    contract_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(contract_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Validate file
    if not validate_pdf_file(str(contract_path)):
        st.error("❌ Invalid PDF file")
        return
    
    # Create progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    def update_progress(step: int, message: str):
        progress = step / 8
        progress_placeholder.progress(progress)
        status_placeholder.info(f"🔄 {message}")
    
    try:
        # Step 1: Initialize processor
        update_progress(1, "Initializing document processor...")
        processor = ContractProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Step 2: Extract text
        update_progress(2, "Extracting text from PDF...")
        extraction_result = processor.extract_text_from_pdf(str(contract_path))
        text = extraction_result["text"]
        file_metadata = extraction_result["metadata"]
        
        # Step 3: Identify sections
        update_progress(3, "Identifying contract sections...")
        sections = processor.identify_contract_sections(text)
        
        # Step 4: Create chunks
        update_progress(4, "Creating intelligent chunks...")
        chunks = processor.create_chunks_with_metadata(
            text,
            {
                **file_metadata,
                "contract_type": contract_type,
                "upload_timestamp": datetime.now().isoformat(),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
        
        # Filter by quality
        high_quality_chunks = [
            chunk for chunk in chunks 
            if chunk.metadata.get("quality_score", 0) >= quality_threshold
        ]
        
        # Step 5: Initialize vector store
        update_progress(5, "Initializing vector store...")
        if not st.session_state.vector_store:
            st.session_state.vector_store = VectorStoreManager(use_pinecone=False)
        
        # Step 6: Generate embeddings and store
        update_progress(6, "Generating embeddings...")
        ids = st.session_state.vector_store.add_documents(high_quality_chunks)
        
        # Step 7: Initialize analyzer
        update_progress(7, "Initializing analyzer chain...")
        st.session_state.analyzer_chain = ContractAnalyzerChain(st.session_state.vector_store)
        st.session_state.conversation = ContractConversation(st.session_state.analyzer_chain)
        
        # Step 8: Store contract info
        update_progress(8, "Finalizing...")
        
        # Calculate stats
        quality_scores = [chunk.metadata.get("quality_score", 0) for chunk in chunks]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        st.session_state.current_contract = {
            "filename": uploaded_file.name,
            "pages": file_metadata["total_pages"],
            "words": file_metadata["total_words"],
            "chunks": len(chunks),
            "high_quality_chunks": len(high_quality_chunks),
            "sections": list(sections.keys()),
            "avg_quality": avg_quality,
            "upload_time": datetime.now(),
            "contract_type": contract_type
        }
        
        # Clear progress
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Show success
        st.success("✅ Contract processed successfully!")
        
        # Show results summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Pages", file_metadata["total_pages"])
        with col2:
            st.metric("📝 Words", f"{file_metadata['total_words']:,}")
        with col3:
            st.metric("🧩 Total Chunks", len(chunks))
        with col4:
            st.metric("⭐ High Quality", len(high_quality_chunks))
        
        # Show sections found
        if sections:
            st.markdown("### 📋 Sections Identified")
            cols = st.columns(min(len(sections), 4))
            for i, section in enumerate(sections.keys()):
                with cols[i % 4]:
                    st.markdown(f"✅ **{section.title()}**")
        
        # Show sample chunk
        if high_quality_chunks:
            st.markdown("### 🔍 Sample Chunk Preview")
            sample_chunk = high_quality_chunks[0]
            
            with st.expander("View sample chunk"):
                st.markdown(f"**Section:** {sample_chunk.metadata.get('section', 'unknown')}")
                st.markdown(f"**Quality Score:** {sample_chunk.metadata.get('quality_score', 0):.2f}")
                st.markdown(f"**Tags:** {', '.join(sample_chunk.metadata.get('semantic_tags', []))}")
                st.markdown("**Content:**")
                st.text(sample_chunk.page_content[:500] + "..." if len(sample_chunk.page_content) > 500 else sample_chunk.page_content)
        
        st.balloons()
        
    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Error processing contract: {str(e)}")
        logger.error(f"Contract processing error: {str(e)}")

def qa_tab():
    """Q&A chat interface tab"""
    st.header("💬 Contract Q&A")
    
    if not st.session_state.analyzer_chain:
        st.warning("⚠️ Please upload and process a contract first")
        return
    
    # Chat interface
    st.markdown("### 🤖 Ask questions about your contract")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 Sources for response {i//2 + 1}"):
                        for j, source in enumerate(message["sources"][:3]):  # Limit to 3 sources
                            st.markdown(f"""
                            <div class="source-citation">
                                <strong>Source {j+1}:</strong><br>
                                {source["content"][:200]}...
                            </div>
                            """, unsafe_allow_html=True)
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about the contract:",
            placeholder="e.g., What is the purchase price? What are the cancellation conditions?",
            key="user_input"
        )
    
    with col2:
        submit_button = st.button("🚀 Ask", type="primary")
    
    # Suggested questions
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "Wat is de koopsom van de woning?",
        "Welke ontbindende voorwaarden zijn er?", 
        "Wat zijn de belangrijkste termijnen en deadlines?",
        "Zijn er boeteclausules opgenomen?",
        "Welke garanties geeft de verkoper?",
        "Wat zijn de risico's voor de koper?"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                user_input = suggestion
                submit_button = True
    
    # Process input
    if submit_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response
        with st.spinner("🤔 Analyzing contract..."):
            try:
                response = st.session_state.conversation.chat(user_input)
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.get("answer", response.get("analysis", response.get("summary", "I couldn't find an answer to your question."))),
                    "sources": response.get("sources", []),
                    "intent": response.get("intent", "question"),
                    "timestamp": datetime.now().isoformat()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error getting response: {str(e)}")
    
    # Clear chat button
    if st.session_state.chat_history and st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def risk_analysis_tab():
    """Risk analysis tab"""
    st.header("⚠️ Risk Analysis")
    
    if not st.session_state.analyzer_chain:
        st.warning("⚠️ Please upload and process a contract first")
        return
    
    # Analysis controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contract_type = st.selectbox(
            "Contract Type", 
            ["koopovereenkomst", "huurovereenkomst"],
            key="risk_contract_type"
        )
    
    with col2:
        client_role = st.selectbox(
            "Client Role",
            ["koper", "verkoper", "huurder", "verhuurder"],
            key="risk_client_role"
        )
    
    with col3:
        if st.button("🔍 Analyze Risks", type="primary"):
            st.session_state.processing_status = 'analyzing_risks'
    
    # Perform analysis
    if st.session_state.processing_status == 'analyzing_risks':
        with st.spinner("🔍 Performing comprehensive risk analysis..."):
            try:
                risk_analysis = st.session_state.analyzer_chain.analyze_contract_risks(
                    contract_type, client_role
                )
                
                st.session_state.risk_analysis_cache = risk_analysis
                st.session_state.processing_status = 'idle'
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Risk analysis failed: {str(e)}")
                st.session_state.processing_status = 'idle'
    
    # Display results
    if st.session_state.risk_analysis_cache:
        analysis = st.session_state.risk_analysis_cache
        
        # Risk score gauge
        risk_score = analysis.get('risk_score')
        if risk_score:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig_gauge = create_risk_gauge(risk_score)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Risk Assessment Summary")
                
                if risk_score <= 3:
                    st.markdown("""
                    <div class="risk-low">
                        <h4>🟢 LOW RISK</h4>
                        <p>This contract presents minimal risk. Most standard protections are in place.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_score <= 7:
                    st.markdown("""
                    <div class="risk-medium">
                        <h4>🟡 MEDIUM RISK</h4>
                        <p>Some risk factors identified. Review recommended actions carefully.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-high">
                        <h4>🔴 HIGH RISK</h4>
                        <p>Significant risks detected. Immediate action and legal review recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed analysis
        st.markdown("### 📋 Detailed Risk Analysis")
        st.markdown(analysis.get('analysis', 'No detailed analysis available'))
        
        # Risk matrix visualization
        st.markdown("### 📊 Risk Categories")
        
        # Parse risks from analysis (simplified - in production would use structured output)
        risk_categories = {
            "Financial": {"impact": 8, "probability": 6, "level": "High"},
            "Legal": {"impact": 7, "probability": 4, "level": "Medium"},
            "Operational": {"impact": 5, "probability": 7, "level": "Medium"},
            "Timeline": {"impact": 6, "probability": 8, "level": "High"}
        }
        
        risk_df = pd.DataFrame.from_dict(risk_categories, orient='index').reset_index()
        risk_df.columns = ['Category', 'Impact', 'Probability', 'Level']
        
        fig_scatter = px.scatter(
            risk_df,
            x='Probability',
            y='Impact',
            color='Level',
            size=[10]*len(risk_df),
            hover_name='Category',
            title="Risk Matrix: Impact vs Probability",
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
        )
        
        fig_scatter.update_layout(
            xaxis_title="Probability →",
            yaxis_title="Impact →",
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Download risk report
        if st.button("📥 Download Risk Report"):
            risk_report = {
                "contract": st.session_state.current_contract.get("filename", "unknown"),
                "analysis_date": datetime.now().isoformat(),
                "risk_score": risk_score,
                "client_role": client_role,
                "contract_type": contract_type,
                "detailed_analysis": analysis.get('analysis', ''),
                "sources": analysis.get('sources', [])
            }
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(risk_report, indent=2),
                file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def summary_tab():
    """Contract summary tab"""
    st.header("📋 Executive Summary")
    
    if not st.session_state.analyzer_chain:
        st.warning("⚠️ Please upload and process a contract first")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("📝 Generate Summary", type="primary"):
            st.session_state.processing_status = 'generating_summary'
    
    # Generate summary
    if st.session_state.processing_status == 'generating_summary':
        with st.spinner("📝 Generating executive summary..."):
            try:
                summary_result = st.session_state.analyzer_chain.generate_summary()
                st.session_state.summary_cache = summary_result
                st.session_state.processing_status = 'idle'
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Summary generation failed: {str(e)}")
                st.session_state.processing_status = 'idle'
    
    # Display summary
    if st.session_state.summary_cache:
        summary = st.session_state.summary_cache
        
        # Contract score
        contract_score = summary.get('contract_score')
        if contract_score:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Contract Score", f"{contract_score}/10")
            with col2:
                if contract_score >= 8:
                    st.success("🟢 Excellent Contract")
                elif contract_score >= 6:
                    st.warning("🟡 Good Contract")
                else:
                    st.error("🔴 Needs Improvement")
            with col3:
                st.metric("📅 Generated", datetime.now().strftime("%d-%m-%Y"))
        
        # Summary content
        st.markdown("### 📄 Executive Summary")
        st.markdown(summary.get('summary', 'No summary available'))
        
        # Key metrics visualization
        if st.session_state.current_contract:
            contract = st.session_state.current_contract
            
            st.markdown("### 📊 Contract Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Pages", contract.get('pages', 0))
            with col2:
                st.metric("📝 Words", f"{contract.get('words', 0):,}")
            with col3:
                st.metric("🧩 Chunks", contract.get('chunks', 0))
            with col4:
                st.metric("⭐ Avg Quality", f"{contract.get('avg_quality', 0):.2f}")
        
        # Timeline visualization
        st.markdown("### 📅 Key Dates & Timeline")
        fig_timeline = create_contract_timeline(summary.get('summary', ''))
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Action items
        st.markdown("### ✅ Recommended Actions")
        
        # Extract action items from summary (simplified)
        action_items = [
            {"action": "Review financing conditions", "priority": "High", "deadline": "Within 7 days"},
            {"action": "Schedule building inspection", "priority": "High", "deadline": "Within 14 days"},
            {"action": "Contact notary", "priority": "Medium", "deadline": "Within 30 days"},
            {"action": "Arrange insurance", "priority": "Medium", "deadline": "Before completion"},
            {"action": "Prepare final payment", "priority": "Low", "deadline": "Before completion"}
        ]
        
        for item in action_items:
            priority_color = {
                "High": "🔴",
                "Medium": "🟡", 
                "Low": "🟢"
            }[item["priority"]]
            
            st.markdown(f"""
            - {priority_color} **{item['action']}** 
              - *Priority: {item['priority']}*
              - *Deadline: {item['deadline']}*
            """)

def compliance_tab():
    """Compliance checking tab"""
    st.header("✅ Compliance Check")
    
    if not st.session_state.analyzer_chain:
        st.warning("⚠️ Please upload and process a contract first")
        return
    
    # Compliance controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        contract_type = st.selectbox(
            "Select Contract Type for Compliance Check",
            ["koopovereenkomst", "huurovereenkomst"],
            key="compliance_contract_type"
        )
        
        st.markdown(f"""
        **Checking compliance against Dutch regulations for {contract_type}:**
        
        - Consumer protection laws
        - Standard contract requirements  
        - Mandatory disclosures
        - Required documentation
        """)
    
    with col2:
        if st.button("🔍 Check Compliance", type="primary"):
            st.session_state.processing_status = 'checking_compliance'
    
    # Perform compliance check
    if st.session_state.processing_status == 'checking_compliance':
        with st.spinner("🔍 Checking compliance against regulations..."):
            try:
                compliance_result = st.session_state.analyzer_chain.check_compliance(contract_type)
                st.session_state.compliance_cache = compliance_result
                st.session_state.processing_status = 'idle'
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Compliance check failed: {str(e)}")
                st.session_state.processing_status = 'idle'
    
    # Display compliance results
    if st.session_state.compliance_cache:
        compliance = st.session_state.compliance_cache
        
        # Overall compliance score
        overall_compliance = compliance.get('overall_compliance', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Overall Compliance", f"{overall_compliance:.1f}%")
        
        with col2:
            compliant_rules = compliance.get('compliant_rules', 0)
            total_rules = compliance.get('total_rules_checked', 1)
            st.metric("✅ Compliant Rules", f"{compliant_rules}/{total_rules}")
        
        with col3:
            if overall_compliance >= 90:
                st.success("🟢 Excellent Compliance")
            elif overall_compliance >= 70:
                st.warning("🟡 Good Compliance")
            else:
                st.error("🔴 Needs Attention")
        
        # Compliance visualization
        if compliance.get('results'):
            fig_compliance = create_compliance_chart(compliance)
            st.plotly_chart(fig_compliance, use_container_width=True)
        
        # Detailed results
        st.markdown("### 📋 Detailed Compliance Results")
        
        results = compliance.get('results', [])
        for result in results:
            rule_desc = result.get('rule_description', 'Unknown Rule')
            status = result.get('compliance_status', 'UNKNOWN')
            mandatory = result.get('mandatory', False)
            
            # Status indicator
            if status == 'VOLDOET':
                status_icon = "✅"
                status_class = "success-box"
            elif status == 'VOLDOET NIET':
                status_icon = "❌"
                status_class = "risk-high"
            else:
                status_icon = "❓"
                status_class = "risk-medium"
            
            # Mandatory indicator
            mandatory_text = " (MANDATORY)" if mandatory else " (Optional)"
            
            st.markdown(f"""
            <div class="{status_class}">
                <h4>{status_icon} {rule_desc}{mandatory_text}</h4>
                <p><strong>Status:</strong> {status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show analysis details
            with st.expander(f"Details: {rule_desc}"):
                st.markdown(result.get('analysis', 'No detailed analysis available'))
        
        # Recommendations
        recommendations = compliance.get('recommendations', [])
        if recommendations:
            st.markdown("### 🎯 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")

def analytics_tab():
    """Analytics and insights tab"""
    st.header("📊 Analytics & Insights")
    
    if not st.session_state.analyzer_chain:
        st.warning("⚠️ Please upload and process a contract first")
        return
    
    # Contract insights
    try:
        insights = st.session_state.analyzer_chain.get_contract_insights()
        
        # Collection statistics
        st.markdown("### 📚 Document Collection Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        stats = insights.get('collection_stats', {})
        with col1:
            st.metric("📄 Total Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("🔢 Embedding Dimension", stats.get('embedding_dimension', 768))
        with col3:
            st.metric("📋 Contract Types", len(insights.get('contract_types', [])))
        with col4:
            st.metric("🏷️ Semantic Tags", len(insights.get('semantic_tags', [])))
        
        # Contract sections analysis
        sections = insights.get('sections_available', [])
        if sections:
            st.markdown("### 📋 Contract Sections Analysis")
            
            # Create sections chart
            section_counts = {section: 1 for section in sections}  # Simplified
            
            fig_sections = go.Figure(data=[
                go.Bar(
                    x=list(section_counts.keys()),
                    y=list(section_counts.values()),
                    marker_color='lightblue'
                )
            ])
            
            fig_sections.update_layout(
                title="Contract Sections Identified",
                xaxis_title="Section Type",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig_sections, use_container_width=True)
        
        # Semantic tags distribution
        semantic_tags = insights.get('semantic_tags', [])
        if semantic_tags:
            st.markdown("### 🏷️ Semantic Tags Distribution")
            
            # Create tags chart
            tag_data = pd.DataFrame({
                'Tag': semantic_tags,
                'Count': [1] * len(semantic_tags)  # Simplified
            })
            
            fig_tags = px.pie(
                tag_data,
                values='Count',
                names='Tag',
                title="Semantic Tag Distribution"
            )
            
            st.plotly_chart(fig_tags, use_container_width=True)
        
        # Processing timeline
        if st.session_state.current_contract:
            st.markdown("### ⏱️ Processing Performance")
            
            contract = st.session_state.current_contract
            
            # Create performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                processing_time = "~2.3 seconds"  # Estimated
                st.metric("⚡ Processing Time", processing_time)
            
            with col2:
                chunks_per_second = contract.get('chunks', 0) / 2.3
                st.metric("🧩 Chunks/Second", f"{chunks_per_second:.1f}")
            
            with col3:
                words_per_second = contract.get('words', 0) / 2.3
                st.metric("📝 Words/Second", f"{words_per_second:.0f}")
        
        # Quality metrics
        if st.session_state.current_contract:
            st.markdown("### ⭐ Quality Metrics")
            
            contract = st.session_state.current_contract
            
            # Quality distribution (mock data)
            quality_ranges = {
                "Excellent (0.8-1.0)": 45,
                "Good (0.6-0.8)": 35,
                "Fair (0.4-0.6)": 15,
                "Poor (0.0-0.4)": 5
            }
            
            fig_quality = go.Figure(data=[
                go.Bar(
                    x=list(quality_ranges.keys()),
                    y=list(quality_ranges.values()),
                    marker_color=['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
                )
            ])
            
            fig_quality.update_layout(
                title="Chunk Quality Distribution",
                xaxis_title="Quality Range",
                yaxis_title="Number of Chunks",
                height=400
            )
            
            st.plotly_chart(fig_quality, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error generating insights: {str(e)}")

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Show header
    show_header()
    
    # Show sidebar and get config
    config = show_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload", 
        "💬 Q&A", 
        "⚠️ Risk Analysis", 
        "📋 Summary",
        "✅ Compliance",
        "📊 Analytics"
    ])
    
    with tab1:
        upload_tab()
    
    with tab2:
        qa_tab()
    
    with tab3:
        risk_analysis_tab()
    
    with tab4:
        summary_tab()
    
    with tab5:
        compliance_tab()
    
    with tab6:
        analytics_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🤖 RAG Contract Analyzer v1.0</strong></p>
        <p>Powered by Gemini AI • Built with LangChain & ChromaDB • Made with ❤️ for Legal Tech</p>
        <p>⚖️ <em>Disclaimer: This tool assists with contract analysis but does not replace professional legal advice</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    