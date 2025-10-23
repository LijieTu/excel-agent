"""
Main Streamlit application for the Intelligent Excel Agent.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import uuid
import logging
from typing import Dict, List, Any, Optional
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from excel_processor import ExcelProcessor
from nlp_agent import NLPAgent, AnalysisIntent
from code_generator import CodeGenerator
from voice_handler import VoiceHandler
from data_tracer import DataTracer
from utils.file_utils import get_available_files, validate_excel_file
from utils.visualization import VisualizationHelper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Intelligent Excel Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'excel_processor' not in st.session_state:
    st.session_state.excel_processor = ExcelProcessor()
if 'nlp_agent' not in st.session_state:
    st.session_state.nlp_agent = NLPAgent()
if 'code_generator' not in st.session_state:
    st.session_state.code_generator = CodeGenerator()
if 'voice_handler' not in st.session_state:
    st.session_state.voice_handler = VoiceHandler()
if 'data_tracer' not in st.session_state:
    st.session_state.data_tracer = DataTracer()
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'current_sheet' not in st.session_state:
    st.session_state.current_sheet = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Intelligent Excel Agent</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to the Intelligent Excel Agent!</strong><br>
        This system can understand natural language queries and automatically perform data analysis on Excel files.
        You can ask questions like "Help me analyze sales trends across regions" and the system will generate
        the appropriate Python code and visualizations.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file selection
    with st.sidebar:
        st.header("📁 File Management")
        
        # File selection
        available_files = get_available_files()
        if available_files:
            file_options = {f["name"]: f["path"] for f in available_files}
            
            # Find sales_data.xlsx index for default selection
            default_index = 0
            for i, name in enumerate(file_options.keys()):
                if "sales_data" in name.lower():
                    default_index = i
                    break
            
            selected_file_name = st.selectbox(
                "Select Excel File:",
                options=list(file_options.keys()),
                index=default_index
            )
            selected_file_path = file_options[selected_file_name]
            
            # Load file if changed
            if st.session_state.current_file != selected_file_path:
                load_excel_file(selected_file_path)
                st.session_state.current_file = selected_file_path
            
            # Sheet selection
            if st.session_state.current_file:
                file_summary = st.session_state.excel_processor.get_file_summary(selected_file_path)
                sheet_names = file_summary['sheets']
                
                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox(
                        "Select Sheet:",
                        options=sheet_names,
                        index=0
                    )
                    st.session_state.current_sheet = selected_sheet
                else:
                    st.session_state.current_sheet = sheet_names[0]
                
                # Display file info
                st.markdown("### File Information")
                st.write(f"**File:** {file_summary['file_path']}")
                st.write(f"**Sheets:** {file_summary['total_sheets']}")
                
                for sheet_name, details in file_summary['sheet_details'].items():
                    st.write(f"**{sheet_name}:** {details['shape'][0]} rows × {details['shape'][1]} columns")
        else:
            st.warning("No Excel files found in the data directory.")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Voice Input", "💬 Text Query", "📊 Results", "🔍 Data Traceability"])
    
    with tab1:
        voice_input_tab()
    
    with tab2:
        text_query_tab()
    
    with tab3:
        results_tab()
    
    with tab4:
        traceability_tab()

def load_excel_file(file_path: str):
    """Load Excel file and update session state."""
    try:
        with st.spinner("Loading Excel file..."):
            processed_sheets = st.session_state.excel_processor.load_excel_file(file_path)
            st.success(f"✅ Successfully loaded {len(processed_sheets)} sheet(s)")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

def voice_input_tab():
    """Voice input tab functionality."""
    st.markdown('<h2 class="section-header">🎤 Voice Input</h2>', unsafe_allow_html=True)
    
    # Voice input interface
    transcript = st.session_state.voice_handler.create_voice_input_ui()
    
    if transcript:
        st.session_state.voice_transcript = transcript
        st.markdown(f"""
        <div class="success-box">
            <strong>Voice Input Received:</strong><br>
            "{transcript}"
        </div>
        """, unsafe_allow_html=True)
        
        # Process the voice input
        if st.button("🔍 Analyze Voice Input"):
            process_query(transcript, "voice")

def text_query_tab():
    """Text query tab functionality."""
    st.markdown('<h2 class="section-header">💬 Text Query</h2>', unsafe_allow_html=True)
    
    # Text input
    query = st.text_area(
        "Enter your analysis query:",
        placeholder="e.g., Help me analyze sales trends across regions",
        height=100
    )
    
    # Example queries
    st.markdown("### 💡 Example Queries")
    example_queries = [
        "Show me the total sales by region",
        "What are the average salaries by department?",
        "Create a trend chart of sales over time",
        "Which products have the highest sales?",
        "Show me the correlation between salary and experience"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"📝 {example[:30]}...", key=f"example_{i}"):
                st.session_state.example_query = example
                st.rerun()
    
    # Use example query if selected
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        st.text_area("Enter your analysis query:", value=query, height=100, key="query_input")
        del st.session_state.example_query
    
    # Process query button
    if st.button("🔍 Analyze Query", type="primary"):
        if query.strip():
            process_query(query, "text")
        else:
            st.warning("Please enter a query to analyze.")

def process_query(query: str, input_type: str):
    """Process user query and generate analysis."""
    if not st.session_state.current_file:
        st.error("❌ Please select an Excel file first.")
        return
    
    try:
        # Get current data
        processed_sheets = st.session_state.excel_processor.processed_files[st.session_state.current_file]
        current_sheet = st.session_state.current_sheet
        df = processed_sheets[current_sheet]
        
        # Parse query
        with st.spinner("Parsing query..."):
            # Force refresh the NLP agent to ensure we have the latest version
            import importlib
            import sys
            
            # Clear any cached modules
            modules_to_reload = ['nlp_agent', 'code_generator']
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
            
            from nlp_agent import NLPAgent
            nlp_agent = NLPAgent()
            intent = nlp_agent.parse_query(query, list(df.columns))
            
            # Debug: Show the parsed intent
            st.write(f"**Parsed Intent:**")
            st.write(f"- Analysis Type: {intent.analysis_type}")
            st.write(f"- Target Columns: {intent.target_columns}")
            st.write(f"- Group By Columns: {intent.group_by_columns}")
            
            # Show OpenAI error if any
            if hasattr(intent, 'openai_error') and intent.openai_error:
                st.warning(f"⚠️ OpenAI Enhancement Issue: {intent.openai_error}")
            
            # Force clear session state cache
            if 'nlp_agent' in st.session_state:
                del st.session_state['nlp_agent']
            if 'code_generator' in st.session_state:
                del st.session_state['code_generator']
            
            # Reinitialize the agents
            st.session_state.nlp_agent = NLPAgent()
            st.session_state.code_generator = CodeGenerator()
        
        # Validate intent
        if not st.session_state.nlp_agent.validate_intent(intent, list(df.columns)):
            st.error("❌ The query references columns that don't exist in the data.")
            return
        
        # Generate code
        with st.spinner("Generating analysis code..."):
            # Clear any cached code and regenerate
            generated_code = st.session_state.code_generator.generate_code(
                intent, st.session_state.current_file, current_sheet
            )
            
            # Debug: Show the generated code
            st.code(generated_code, language='python')
        
        # Execute code
        with st.spinner("Executing analysis..."):
            start_time = time.time()
            
            # Create execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'make_subplots': make_subplots
            }
            
            # Execute the generated code
            exec(generated_code, exec_globals)
            
            execution_time = time.time() - start_time
            
            # Get result
            if 'result' in exec_globals:
                result_df = exec_globals['result']
            else:
                result_df = df  # Fallback to original data
            
            # Get figure if created
            figure = None
            if 'fig' in exec_globals:
                figure = exec_globals['fig']
        
        # Trace the analysis
        analysis_id = str(uuid.uuid4())
        trace = st.session_state.data_tracer.trace_analysis(
            analysis_id=analysis_id,
            query=query,
            file_path=st.session_state.current_file,
            sheet_name=current_sheet,
            intent=intent,
            df=df,
            generated_code=generated_code,
            execution_time=execution_time,
            result_df=result_df,
            success=True
        )
        
        # Store in session state
        analysis_result = {
            'id': analysis_id,
            'query': query,
            'input_type': input_type,
            'intent': intent,
            'generated_code': generated_code,
            'result_df': result_df,
            'figure': figure,
            'execution_time': execution_time,
            'trace': trace,
            'timestamp': time.time()
        }
        
        st.session_state.analysis_history.append(analysis_result)
        
        st.success("✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(f"❌ Error processing query: {str(e)}")
        
        # Trace failed analysis
        if 'analysis_id' in locals():
            st.session_state.data_tracer.trace_analysis(
                analysis_id=analysis_id,
                query=query,
                file_path=st.session_state.current_file,
                sheet_name=current_sheet,
                intent=intent,
                df=df,
                generated_code=generated_code,
                execution_time=0,
                result_df=pd.DataFrame(),
                success=False,
                error_message=str(e)
            )

def results_tab():
    """Results tab functionality."""
    st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.info("No analysis results yet. Please run an analysis first.")
        return
    
    # Select analysis to display
    analysis_options = {
        f"{i+1}. {result['query'][:50]}...": i 
        for i, result in enumerate(st.session_state.analysis_history)
    }
    
    selected_analysis = st.selectbox(
        "Select Analysis to View:",
        options=list(analysis_options.keys()),
        index=len(st.session_state.analysis_history) - 1
    )
    
    analysis_index = analysis_options[selected_analysis]
    analysis = st.session_state.analysis_history[analysis_index]
    
    # Display analysis details
    st.markdown("### 📋 Analysis Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Query", analysis['query'])
    with col2:
        st.metric("Analysis Type", analysis['intent'].analysis_type.value)
    with col3:
        st.metric("Execution Time", f"{analysis['execution_time']:.2f}s")
    
    # Display generated code
    st.markdown("### 💻 Generated Code")
    st.code(analysis['generated_code'], language='python')
    
    # Display results
    st.markdown("### 📊 Results")
    
    if not analysis['result_df'].empty:
        st.dataframe(analysis['result_df'], use_container_width=True)
        
        # Display visualization if available
        if analysis['figure']:
            st.markdown("### 📈 Visualization")
            st.plotly_chart(analysis['figure'], use_container_width=True)
        else:
            # Create default visualization
            if len(analysis['result_df'].columns) >= 2:
                fig = VisualizationHelper.create_chart(
                    analysis['result_df'],
                    chart_type=analysis['intent'].chart_type or 'bar',
                    title=f"Analysis Results: {analysis['query']}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Display data traceability
    st.markdown("### 🔍 Data Traceability")
    trace = analysis['trace']
    
    st.markdown("**Columns Used:**")
    for col in trace.used_columns:
        st.write(f"• **{col.column_name}** ({col.usage_type}) - {col.operation}")
    
    st.markdown(f"**Analysis ID:** {trace.analysis_id}")
    st.markdown(f"**Timestamp:** {trace.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

def traceability_tab():
    """Data traceability tab functionality."""
    st.markdown('<h2 class="section-header">🔍 Data Traceability</h2>', unsafe_allow_html=True)
    
    # Overall summary
    summary = st.session_state.data_tracer.get_analysis_summary()
    
    if summary['total_analyses'] > 0:
        st.markdown("### 📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", summary['total_analyses'])
        with col2:
            st.metric("Success Rate", f"{summary['success_rate']:.1%}")
        with col3:
            st.metric("Unique Files", summary['unique_files'])
        with col4:
            st.metric("Avg Execution Time", f"{summary['average_execution_time']:.2f}s")
        
        # Column usage statistics
        st.markdown("### 📊 Column Usage Statistics")
        
        if st.session_state.current_file:
            usage_stats = st.session_state.data_tracer.get_column_usage_report(
                st.session_state.current_file, st.session_state.current_sheet
            )
            
            if usage_stats:
                usage_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Usage Count': stats['usage_count'],
                        'Usage Types': ', '.join(stats['usage_types']),
                        'Operations': ', '.join(stats['operations']),
                        'Last Used': stats['last_used'].strftime('%Y-%m-%d %H:%M:%S') if stats['last_used'] else 'Never'
                    }
                    for col, stats in usage_stats.items()
                ])
                
                st.dataframe(usage_df, use_container_width=True)
            else:
                st.info("No column usage statistics available for the current file.")
        
        # Export trace data
        st.markdown("### 📤 Export Trace Data")
        if st.button("Export Trace Data to JSON"):
            export_path = "trace_data.json"
            st.session_state.data_tracer.export_trace_data(export_path)
            st.success(f"Trace data exported to {export_path}")
    
    else:
        st.info("No analysis history available. Run some analyses to see traceability information.")

if __name__ == "__main__":
    try:
        # Validate configuration
        Config.validate_config()
        main()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.markdown("""
        <div class="error-box">
            <strong>Setup Required:</strong><br>
            1. Copy <code>env_example.txt</code> to <code>.env</code><br>
            2. Add your OpenAI API key to the <code>.env</code> file<br>
            3. Restart the application
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


