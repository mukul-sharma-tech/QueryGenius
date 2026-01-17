import streamlit as st
from utils import (
    upload_to_mongo,
    fetch_schema,
    get_mongo_client,
    trend_prediction_model,
    execute_mongo_query,
    get_all_collections,
    get_collection_stats,
    get_collection_data,
    delete_collection,
    search_in_collection,
    get_collection_aggregations,
    test_mongo_connection,
    get_database_info,
    trend_prediction_model,
    anomaly_detection,
    correlation_analysis,
    generate_ml_report
)
import numpy as np
import google.generativeai as genai
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# Set page config
st.set_page_config(
    page_title="QueryGenius - NL to MongoDB",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    
    /* Header styling */
    .stTitle {
        color: #2c3e50;
        font-weight: 700;
        font-size: 2.5rem;
        padding-bottom: 10px;
        border-bottom: 3px solid #3498db;
        margin-bottom: 20px;
    }
    
    /* Card-like containers */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        margin-bottom: 25px;
        border: none;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1a252f 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
    }
    
    /* Primary button */
    .primary-button > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        color: #7f8c8d;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #3498db !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        color: white;
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #e8f4f8;
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        background-color: #d4edf7;
        border-color: #2980b9;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: none;
        color: #155724;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: none;
        color: #721c24;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: none;
        color: #0c5460;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-success {
        background-color: #28a745;
        color: white;
    }
    
    .badge-warning {
        background-color: #ffc107;
        color: #212529;
    }
    
    .badge-info {
        background-color: #17a2b8;
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #3498db;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #3498db, transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = None
if 'all_collections' not in st.session_state:
    st.session_state.all_collections = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'database_info' not in st.session_state:
    st.session_state.database_info = {}

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: white; font-size: 28px; margin-bottom: 10px;'>🚀 QueryGenius</h1>
        <p style='color: #bdc3c7; font-size: 14px; margin-bottom: 30px;'>Natural Language to MongoDB Query Engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("---")
    page = st.radio(
        "📌 **NAVIGATION**",
        ["📤 Add Files", "🔍 Explore Data", "🤖 ML Analysis", "⚙️ Database Settings"],
        key="nav",
        label_visibility="collapsed"
    )
    
    # Database Status
    st.markdown("---")
    st.markdown("### 📊 Database Status")
    
    # Test connection
    if st.button("🔄 Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            connection_status = test_mongo_connection()
            if connection_status.get("connected", False):
                st.success("✅ Connected!")
                st.session_state.database_info = connection_status
            else:
                st.error("❌ Connection Failed")
    
    # Get all collections if connected
    try:
        collections = get_all_collections()
        st.session_state.all_collections = collections
        
        if collections:
            st.success(f"✅ **{len(collections)}** collections found")
            
            # Collection selector
            selected_collection = st.selectbox(
                "**Select Collection**",
                ["-- Select --"] + collections,
                key="sidebar_collection"
            )
            
            if selected_collection != "-- Select --":
                st.session_state.current_collection = selected_collection
                
                # Show collection stats
                stats = get_collection_stats(selected_collection)
                if stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📝 Records", f"{stats.get('count', 0):,}")
                    with col2:
                        st.metric("📊 Fields", len(stats.get('fields', [])))
        else:
            st.info("📭 No collections yet")
    except Exception as e:
        st.error("❌ Database error")
    
    # Quick Actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.session_state.current_collection:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Delete", use_container_width=True):
                st.warning(f"Delete collection '{st.session_state.current_collection}'?")
                if st.button("Confirm Delete", type="primary"):
                    if delete_collection(st.session_state.current_collection):
                        st.success("Collection deleted!")
                        st.session_state.all_collections = get_all_collections()
                        st.session_state.current_collection = None
                        st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.all_collections = get_all_collections()
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #95a5a6; font-size: 12px; padding: 20px 0;'>
        <p>QueryGenius v1.0</p>
        <p>Powered by MongoDB + Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

# Main content based on selected page
if page == "📤 Add Files":
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("📤 Upload Data Files")
        st.markdown("Upload CSV or Excel files to create MongoDB collections")
    
    # File Upload Card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.markdown("### 📁 Upload New File")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "**Drag and drop or click to browse**",
                type=["csv", "xlsx"],
                help="Supported formats: CSV, Excel (.xlsx)",
                key="main_file_uploader"
            )
        
        with col2:
            default_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            collection_name = st.text_input(
                "**Collection Name**",
                value=default_name,
                help="Unique name for your MongoDB collection"
            )
        
        if uploaded_file:
            # File preview section
            st.markdown("---")
            st.markdown("### 📄 File Preview")
            
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_preview = pd.read_csv(uploaded_file)
                else:
                    df_preview = pd.read_excel(uploaded_file, engine='openpyxl')
                
                # File info in metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Rows", f"{len(df_preview):,}")
                with col2:
                    st.metric("📈 Columns", len(df_preview.columns))
                with col3:
                    file_size_kb = uploaded_file.size / 1024
                    st.metric("💾 Size", f"{file_size_kb:.1f} KB")
                with col4:
                    st.metric("📝 File Type", uploaded_file.name.split('.')[-1].upper())
                
                # Data preview
                st.markdown("#### Sample Data (First 10 rows)")
                st.dataframe(df_preview.head(10), use_container_width=True)
                
                # Column info
                with st.expander("📋 Column Information", expanded=True):
                    col_info = pd.DataFrame({
                        'Column': df_preview.columns,
                        'Data Type': df_preview.dtypes.astype(str).values,
                        'Non-Null Count': df_preview.notnull().sum().values,
                        'Null Count': df_preview.isnull().sum().values
                    })
                    st.dataframe(col_info, use_container_width=True, hide_index=True)
                
                # Upload button
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🚀 **Upload to Database**", type="primary", use_container_width=True, 
                               help="Upload data to MongoDB Atlas"):
                        with st.spinner("Uploading to MongoDB..."):
                            progress_bar = st.progress(0)
                            
                            # Simulate progress
                            for i in range(100):
                                progress_bar.progress(i + 1)
                            
                            count = upload_to_mongo(uploaded_file, collection_name)
                            
                            if count > 0:
                                st.success(f"✅ Successfully uploaded **{count:,}** records to collection: `{collection_name}`")
                                st.balloons()
                                
                                # Update session state
                                st.session_state.uploaded_files.append({
                                    'filename': uploaded_file.name,
                                    'collection': collection_name,
                                    'rows': count,
                                    'timestamp': datetime.now(),
                                    'size_kb': file_size_kb
                                })
                                st.session_state.all_collections = get_all_collections()
                                
                                # Show schema
                                with st.expander("🗃️ Collection Schema", expanded=True):
                                    schema = fetch_schema(collection_name)
                                    if schema:
                                        st.json(schema, expanded=False)
                                        
                                        # Show field badges
                                        st.markdown("**Fields:**")
                                        for field in schema.keys():
                                            st.markdown(f'<span class="badge badge-info">{field}</span>', 
                                                       unsafe_allow_html=True)
                            else:
                                st.error("❌ Failed to upload data. Please check your connection.")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure the file is in correct format and not corrupted.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload History Section
    if st.session_state.uploaded_files:
        st.markdown("### 📋 Upload History")
        
        history_df = pd.DataFrame(st.session_state.uploaded_files)
        if not history_df.empty:
            # Format timestamp
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display as cards
            cols = st.columns(2)
            for idx, row in history_df.iterrows():
                with cols[idx % 2]:
                    with st.container():
                        st.markdown(f"""
                        <div class="card" style="padding: 15px;">
                            <h4>📁 {row['filename']}</h4>
                            <p><strong>Collection:</strong> <code>{row['collection']}</code></p>
                            <p><strong>Records:</strong> {row['rows']:,}</p>
                            <p><strong>Size:</strong> {row.get('size_kb', 0):.1f} KB</p>
                            <p><strong>Uploaded:</strong> {row['timestamp']}</p>
                        </div>
                        """, unsafe_allow_html=True)

elif page == "🔍 Explore Data":
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Explore & Query Data")
        st.markdown("Query your data using natural language or explore visually")
    
    # Collection Selector
    if st.session_state.all_collections:
        selected_collection = st.selectbox(
            "**Select Collection to Explore**",
            st.session_state.all_collections,
            index=0 if not st.session_state.current_collection or 
                      st.session_state.current_collection not in st.session_state.all_collections 
                   else st.session_state.all_collections.index(st.session_state.current_collection),
            key="main_collection_selector"
        )
        st.session_state.current_collection = selected_collection
        
        if selected_collection:
            # Collection Stats
            stats = get_collection_stats(selected_collection)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Records</h3>
                    <div class="value">{stats.get('count', 0):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fields</h3>
                    <div class="value">{len(stats.get('fields', []))}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                size_mb = stats.get('size_bytes', 0) / (1024 * 1024)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Size</h3>
                    <div class="value">{size_mb:.2f} MB</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Indexes</h3>
                    <div class="value">{stats.get('indexes', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📋 View Data", "💬 NL Query", "🔍 Search", "📊 Analytics"])
            
            with tab1:
                # View all data
                st.markdown(f"### 📊 Data Preview: `{selected_collection}`")
                
                # Load data
                df = get_collection_data(selected_collection, limit=1000)
                
                if not df.empty:
                    # Show data with filters
                    st.dataframe(df, use_container_width=True, height=500)
                    
                    # Show data info
                    with st.expander("📈 Data Summary", expanded=False):
                        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                        
                        # Numeric columns summary
                        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_cols) > 0:
                            st.markdown("**Numeric Columns:**")
                            for col in numeric_cols:
                                st.write(f"- **{col}:** Min={df[col].min():.2f}, Max={df[col].max():.2f}, "
                                        f"Mean={df[col].mean():.2f}, Std={df[col].std():.2f}")
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Dataset",
                            data=csv,
                            file_name=f"{selected_collection}_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("No data found in this collection.")
            
            with tab2:
                # Natural Language Query
                st.markdown(f"### 💬 Natural Language Query")
                st.markdown("Ask questions about your data in plain English")
                
                # Query examples
                with st.expander("📚 Query Examples", expanded=False):
                    examples = [
                        "Show top 5 employees by marks",
                        "Find all people in Mumbai",
                        "Average salary by city",
                        "Count employees in each location",
                        "Employees with salary above 50000",
                        "Sort by salary descending",
                        "Group by job_location and show average marks",
                        "Find duplicate emails",
                        "Highest and lowest marks",
                        "Distribution of salaries"
                    ]
                    
                    cols = st.columns(2)
                    for idx, example in enumerate(examples):
                        with cols[idx % 2]:
                            if st.button(f"✨ {example}", key=f"example_{idx}", use_container_width=True):
                                st.session_state.query_text = example
                
                # Query input
                query = st.text_area(
                    "**Enter your query**",
                    value=st.session_state.get('query_text', ''),
                    placeholder="e.g., 'Show top 5 employees by marks in Mumbai'",
                    height=100,
                    key="nl_query_input"
                )
                
                api_key = os.getenv("GOOGLE_API_KEY")
                
                if api_key:
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        if st.button("🚀 **Execute Query**", type="primary", use_container_width=True):
                            st.session_state.run_nl_query = True
                    
                    if st.session_state.get('run_nl_query', False) and query:
                        with st.spinner("🧠 Converting to MongoDB query..."):
                            try:
                                # Configure Google Generative AI
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                
                                # Get schema for prompt
                                schema = fetch_schema(selected_collection)
                                
                                # Create prompt
                                prompt = f"""
                                You are a MongoDB expert. Convert this natural language query to a MongoDB aggregation pipeline.
                                
                                COLLECTION: {selected_collection}
                                SCHEMA: {json.dumps(schema, indent=2) if schema else 'No schema'}
                                QUERY: "{query}"
                                
                                IMPORTANT:
                                1. Return ONLY a valid JSON array
                                2. No markdown, no explanations
                                3. Use $regex with 'i' for text searches
                                4. If unsure, return []
                                
                                RESPONSE:
                                """
                                
                                response = model.generate_content(prompt)
                                raw_pipeline = response.text
                                
                                # Clean response
                                cleaned = raw_pipeline.strip()
                                if cleaned.startswith("```json"):
                                    cleaned = cleaned[7:]
                                if cleaned.startswith("```"):
                                    cleaned = cleaned[3:]
                                if cleaned.endswith("```"):
                                    cleaned = cleaned[:-3]
                                cleaned = cleaned.strip()
                                
                                # Try to parse
                                try:
                                    pipeline = json.loads(cleaned.replace("'", '"'))
                                    
                                    # Show pipeline
                                    with st.expander("🔧 Generated MongoDB Pipeline", expanded=True):
                                        st.code(json.dumps(pipeline, indent=2), language="json")
                                    
                                    # Execute
                                    results = execute_mongo_query(pipeline, selected_collection)
                                    
                                    if results:
                                        df_results = pd.DataFrame(results)
                                        if '_id' in df_results.columns:
                                            df_results = df_results.drop('_id', axis=1)
                                        
                                        st.success(f"✅ Retrieved **{len(df_results)}** records")
                                        st.dataframe(df_results, use_container_width=True)
                                        
                                        # Show stats
                                        if not df_results.empty:
                                            with st.expander("📊 Query Statistics", expanded=False):
                                                numeric_cols = df_results.select_dtypes(include=['int64', 'float64']).columns
                                                for col in numeric_cols:
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric(f"Avg {col}", f"{df_results[col].mean():.2f}")
                                                    with col2:
                                                        st.metric(f"Min {col}", f"{df_results[col].min():.2f}")
                                                    with col3:
                                                        st.metric(f"Max {col}", f"{df_results[col].max():.2f}")
                                        
                                        # Download
                                        csv_results = df_results.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download Results",
                                            csv_results,
                                            f"{selected_collection}_query_results.csv",
                                            "text/csv",
                                            use_container_width=True
                                        )
                                    else:
                                        st.info("No results found for your query.")
                                
                                except json.JSONDecodeError:
                                    st.error("Could not parse the AI response.")
                                    st.code(raw_pipeline, language="text")
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)[:200]}")
                        
                        st.session_state.run_nl_query = False
                else:
                    st.warning("⚠️ Google API key not found. Please set GOOGLE_API_KEY in .env file")
            
            with tab3:
                # Search Tab
                st.markdown("### 🔍 Search in Collection")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_text = st.text_input("Search text", placeholder="Enter search term...")
                with col2:
                    search_field = st.selectbox("Field", ["All Fields"] + stats.get('fields', []))
                
                if search_text:
                    with st.spinner("Searching..."):
                        if search_field == "All Fields":
                            results = search_in_collection(selected_collection, search_text)
                        else:
                            results = search_in_collection(selected_collection, search_text, search_field)
                        
                        if results:
                            df_search = pd.DataFrame(results)
                            st.success(f"Found {len(df_search)} matches")
                            st.dataframe(df_search, use_container_width=True)
                        else:
                            st.info("No matches found")
            
            with tab4:
                # Analytics Tab
                st.markdown("### 📊 Data Analytics")
                
                df = get_collection_data(selected_collection, limit=1000)
                
                if not df.empty:
                    # Numeric Analysis
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    
                    if len(numeric_cols) > 0:
                        st.markdown("#### 🔢 Numeric Analysis")
                        
                        selected_num_col = st.selectbox("Select numeric column", numeric_cols)
                        
                        if selected_num_col:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{df[selected_num_col].mean():.2f}")
                            with col2:
                                st.metric("Median", f"{df[selected_num_col].median():.2f}")
                            with col3:
                                st.metric("Std Dev", f"{df[selected_num_col].std():.2f}")
                            with col4:
                                st.metric("Range", f"{df[selected_num_col].max() - df[selected_num_col].min():.2f}")
                            
                            # Histogram
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.hist(df[selected_num_col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                            ax.set_xlabel(selected_num_col)
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Distribution of {selected_num_col}')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    # Categorical Analysis
                    cat_cols = df.select_dtypes(include=['object']).columns
                    
                    if len(cat_cols) > 0:
                        st.markdown("#### 📊 Categorical Analysis")
                        
                        selected_cat_col = st.selectbox("Select categorical column", cat_cols)
                        
                        if selected_cat_col:
                            value_counts = df[selected_cat_col].value_counts().head(10)
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                value_counts.plot(kind='bar', ax=ax, color='skyblue')
                                ax.set_xlabel(selected_cat_col)
                                ax.set_ylabel('Count')
                                ax.set_title(f'Top 10 Values in {selected_cat_col}')
                                ax.tick_params(axis='x', rotation=45)
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            
                            with col2:
                                st.dataframe(value_counts)
                    
                    # Correlation Matrix
                    if len(numeric_cols) >= 2:
                        st.markdown("#### 🔗 Correlation Matrix")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        corr_matrix = df[numeric_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                        st.pyplot(fig)
                else:
                    st.info("No data available for analytics.")
    else:
        # No collections
        st.info("""
        ## 📭 No Collections Found
        
        To get started:
        1. Go to **📤 Add Files** in the sidebar
        2. Upload a CSV or Excel file
        3. Give it a collection name
        4. Click **Upload to Database**
        
        Then come back here to explore and query your data!
        """)

elif page == "🤖 ML Analysis":
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("🤖 ML Analysis Dashboard")
        st.markdown("Advanced analytics for UIDAI data using Machine Learning")

    # Collection selection for ML analysis
    st.markdown("---")
    st.markdown("### 📊 Select Dataset for Analysis")

    if st.session_state.all_collections:
        selected_collection_ml = st.selectbox(
            "**Choose a collection for ML analysis**",
            ["-- Select Collection --"] + st.session_state.all_collections,
            key="ml_collection"
        )

        if selected_collection_ml != "-- Select Collection --":
            # Get data from collection
            try:
                client = get_mongo_client()
                if client:
                    db = client["nl_mongo_db"]
                    collection = db[selected_collection_ml]

                    # Convert to DataFrame
                    data = list(collection.find({}, {"_id": 0}))
                    df = pd.DataFrame(data)

                    if not df.empty:
                        st.success(f"✅ Loaded {len(df)} records from '{selected_collection_ml}'")

                        # ML Analysis Tabs
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📈 Trend Prediction",
                            "🔍 Anomaly Detection",
                            "📊 Correlation Analysis",
                            "📋 Dashboard",
                            "📄 Report Generation"
                        ])

                        with tab1:
                            st.markdown("### 📈 Trend Prediction Model")
                            st.markdown("Predict future trends using ARIMA time series forecasting")

                            # Select target column
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                target_col = st.selectbox("Select target column for prediction", numeric_cols, key="trend_target")
                                periods = st.slider("Forecast periods", 1, 24, 12, key="forecast_periods")

                                if st.button("🚀 Run Trend Prediction", key="trend_btn"):
                                    with st.spinner("Analyzing trends..."):
                                        trend_results = trend_prediction_model(df, target_col, periods)

                                    if 'error' not in trend_results:
                                        st.success("✅ Trend analysis completed!")

                                        # Plot historical + forecast
                                        fig = go.Figure()

                                        # Historical data
                                        hist_dates = list(trend_results['historical'].keys())
                                        hist_values = list(trend_results['historical'].values())
                                        fig.add_trace(go.Scatter(x=hist_dates, y=hist_values, mode='lines+markers', name='Historical'))

                                        # Forecast
                                        forecast_dates = list(trend_results['forecast'].keys())
                                        forecast_values = list(trend_results['forecast'].values())
                                        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines+markers', name='Forecast', line=dict(dash='dash')))

                                        fig.update_layout(title=f"Trend Prediction for {target_col}", xaxis_title="Date", yaxis_title=target_col)
                                        st.plotly_chart(fig, use_container_width=True)

                                        with st.expander("📊 Model Details"):
                                            st.code(trend_results['model_summary'])
                                    else:
                                        st.error(f"❌ {trend_results['error']}")
                            else:
                                st.warning("No numeric columns found for trend analysis")

                        with tab2:
                            st.markdown("### 🔍 Anomaly Detection")
                            st.markdown("Identify suspicious patterns using Isolation Forest algorithm")

                            contamination = st.slider("Contamination (expected anomaly %)", 0.01, 0.5, 0.1, key="contamination")

                            if st.button("🔍 Detect Anomalies", key="anomaly_btn"):
                                with st.spinner("Detecting anomalies..."):
                                    anomaly_results = anomaly_detection(df, contamination)

                                if 'error' not in anomaly_results:
                                    st.success("✅ Anomaly detection completed!")

                                    # Metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Records", anomaly_results['total_records'])
                                    with col2:
                                        st.metric("Anomalies Found", anomaly_results['anomaly_count'])
                                    with col3:
                                        st.metric("Anomaly %", f"{anomaly_results['anomaly_percentage']:.2f}%")

                                    # Anomalous records
                                    if anomaly_results['anomalous_records']:
                                        st.markdown("### 🚨 Anomalous Records")
                                        anomaly_df = pd.DataFrame(anomaly_results['anomalous_records'])
                                        st.dataframe(anomaly_df, use_container_width=True)

                                        # Visualization
                                        fig = px.scatter(df, x=df.index, y=df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns[0],
                                                       color=df.index.isin(anomaly_df.index) if hasattr(anomaly_df, 'index') else [False]*len(df),
                                                       title="Anomaly Detection Results")
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error(f"❌ {anomaly_results['error']}")

                        with tab3:
                            st.markdown("### 📊 Correlation Analysis")
                            st.markdown("Analyze relationships between different datasets")

                            # Select second dataset for correlation
                            other_collections = [c for c in st.session_state.all_collections if c != selected_collection_ml]
                            if other_collections:
                                corr_collection = st.selectbox("Select second dataset for correlation", ["-- None --"] + other_collections, key="corr_collection")

                                if corr_collection != "-- None --":
                                    method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_method")

                                    if st.button("📊 Analyze Correlation", key="corr_btn"):
                                        with st.spinner("Analyzing correlations..."):
                                            # Get second dataset
                                            corr_data = list(db[corr_collection].find({}, {"_id": 0}))
                                            df2 = pd.DataFrame(corr_data)

                                            correlation_results = correlation_analysis(df, df2, method)

                                        if 'error' not in correlation_results:
                                            st.success("✅ Correlation analysis completed!")

                                            # Display strongest correlations
                                            st.markdown("### 🔗 Strongest Correlations")
                                            for col, corr in correlation_results['strongest_correlations']:
                                                st.write(f"**{col}**: {corr:.3f}")

                                            # Correlation heatmap
                                            corr_matrix = pd.DataFrame(correlation_results['correlation_matrix'])
                                            fig = px.imshow(corr_matrix, title="Correlation Matrix", aspect="auto")
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.error(f"❌ {correlation_results['error']}")
                                else:
                                    st.info("Select a second dataset to perform correlation analysis")
                            else:
                                st.warning("No other collections available for correlation analysis")

                        with tab4:
                            st.markdown("### 📋 ML Dashboard")
                            st.markdown("Comprehensive visualization dashboard")

                            # Quick metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Records", len(df))
                            with col2:
                                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
                            with col3:
                                st.metric("Date Columns", len([col for col in df.columns if 'date' in col.lower()]))
                            with col4:
                                st.metric("Location Columns", len([col for col in df.columns if any(x in col.lower() for x in ['state', 'district', 'pincode'])]))

                            # Data overview plots
                            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                                st.markdown("### 📊 Data Distributions")
                                numeric_df = df.select_dtypes(include=[np.number])
                                fig = px.box(numeric_df, title="Numeric Data Distributions")
                                st.plotly_chart(fig, use_container_width=True)

                            # Time series if date column exists
                            date_cols = [col for col in df.columns if 'date' in col.lower()]
                            if date_cols and len(df.select_dtypes(include=[np.number]).columns) > 0:
                                st.markdown("### 📈 Time Series Trends")
                                df_ts = df.copy()
                                df_ts[date_cols[0]] = pd.to_datetime(df_ts[date_cols[0]], errors='coerce')
                                df_ts = df_ts.dropna(subset=[date_cols[0]])
                                df_ts = df_ts.set_index(date_cols[0])

                                numeric_cols = df_ts.select_dtypes(include=[np.number]).columns[:3]  # Top 3 numeric columns
                                if len(numeric_cols) > 0:
                                    fig = px.line(df_ts, y=numeric_cols, title="Time Series Trends")
                                    st.plotly_chart(fig, use_container_width=True)

                        with tab5:
                            st.markdown("### 📄 Automated Report Generation")
                            st.markdown("Generate comprehensive ML analysis reports")

                            if st.button("📄 Generate Full Report", key="report_btn"):
                                with st.spinner("Generating report..."):
                                    # Run all analyses
                                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                                    trend_results = {}
                                    anomaly_results = {}
                                    correlation_results = {}

                                    if numeric_cols:
                                        trend_results = trend_prediction_model(df, numeric_cols[0], 12)
                                        anomaly_results = anomaly_detection(df, 0.1)

                                        # Try correlation with another dataset if available
                                        other_collections = [c for c in st.session_state.all_collections if c != selected_collection_ml]
                                        if other_collections:
                                            corr_data = list(db[other_collections[0]].find({}, {"_id": 0}))
                                            df2 = pd.DataFrame(corr_data)
                                            correlation_results = correlation_analysis(df, df2, 'pearson')
                                        else:
                                            correlation_results = {"error": "No second dataset available"}

                                    report = generate_ml_report(trend_results, anomaly_results, correlation_results)

                                st.success("✅ Report generated!")
                                st.markdown("### 📋 Analysis Report")
                                st.code(report, language="markdown")

                                # Download button
                                st.download_button(
                                    label="📥 Download Report",
                                    data=report,
                                    file_name=f"ml_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown",
                                    key="download_report"
                                )

                    else:
                        st.warning("Selected collection is empty")
                else:
                    st.error("Database connection failed")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        else:
            st.info("Please select a collection to perform ML analysis")
    else:
        st.warning("No collections available. Please upload data first.")

elif page == "⚙️ Database Settings":
    st.title("⚙️ Database Settings")
    
    # Test Connection
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔗 Connection Status")
        
        if st.button("🔄 Test Database Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                connection_status = test_mongo_connection()
                
                if connection_status.get("connected", False):
                    st.success("✅ **Connected Successfully!**")
                    
                    # Display connection info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Database", connection_status.get("database", "N/A"))
                        st.metric("Collections", connection_status.get("collections_count", 0))
                    with col2:
                        server_info = connection_status.get("server_info", {})
                        st.metric("MongoDB Version", server_info.get("version", "N/A"))
                        st.metric("Host", server_info.get("host", "N/A"))
                    
                    # Show collections
                    if connection_status.get("collections"):
                        st.markdown("**Available Collections:**")
                        for col in connection_status["collections"]:
                            st.markdown(f"- `{col}`")
                else:
                    st.error("❌ **Connection Failed**")
                    st.code(connection_status.get("error", "Unknown error"), language="text")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Database Information
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Database Information")
        
        if st.button("📋 Get Database Info", use_container_width=True):
            with st.spinner("Fetching database information..."):
                db_info = get_database_info()
                
                if db_info:
                    st.success(f"Database: **{db_info.get('database')}**")
                    st.metric("Total Collections", db_info.get("total_collections", 0))
                    
                    # Show collections table
                    collections_data = []
                    for col in db_info.get("collections", []):
                        collections_data.append({
                            "Collection": col.get("name"),
                            "Records": col.get("count", 0),
                            "Fields": len(col.get("fields", [])),
                            "Indexes": col.get("indexes", 0)
                        })
                    
                    if collections_data:
                        st.dataframe(pd.DataFrame(collections_data), use_container_width=True)
                else:
                    st.error("Failed to fetch database information")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System Information
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Python Version", os.sys.version.split()[0])
            st.metric("Streamlit Version", st.__version__)
        with col2:
            try:
                import pymongo
                st.metric("PyMongo Version", pymongo.__version__)
            except:
                st.metric("PyMongo Version", "N/A")
            
            try:
                import pandas as pd
                st.metric("Pandas Version", pd.__version__)
            except:
                st.metric("Pandas Version", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**QueryGenius v1.0**")
with col2:
    st.markdown("Made with ❤️ using Streamlit")
with col3:
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")