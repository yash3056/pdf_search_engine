"""
Streamlit web interface for the PDF search engine.
"""
import streamlit as st
import requests
import time
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import pandas as pd

from config import config


# Configure Streamlit page
st.set_page_config(
    page_title="PDF Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .search-box {
        margin: 2rem 0;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .result-header {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .result-meta {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .highlight {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .confidence-score {
        background-color: #4caf50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class SearchInterface:
    """Main search interface class."""
    
    def __init__(self):
        # Determine API base URL based on environment
        api_host = os.getenv("API_HOST", "localhost")
        if api_host == "pdf-search-api":  # Docker service name
            self.api_base_url = f"http://{api_host}:{config.API_PORT}"
        else:
            self.api_base_url = f"http://localhost:{config.API_PORT}"
        
        self.results_per_page = config.RESULTS_PER_PAGE
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state."""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_results' not in st.session_state:
            st.session_state.current_results = []
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        if 'last_query' not in st.session_state:
            st.session_state.last_query = ""
        if 'index_status' not in st.session_state:
            st.session_state.index_status = None
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check if the API is healthy and get status."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "unreachable", "error": str(e)}
    
    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """Get index statistics from the API."""
        try:
            response = requests.get(f"{self.api_base_url}/index/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None
    
    def search_pdfs(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """Search PDFs using the API."""
        if max_results is None:
            max_results = config.MAX_RESULTS
        
        try:
            response = requests.get(
                f"{self.api_base_url}/search",
                params={"q": query, "max_results": max_results},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type") == "application/json" else response.text
                return {
                    "error": f"Search failed: {error_detail}",
                    "status_code": response.status_code
                }
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def index_pdfs(self) -> Dict[str, Any]:
        """Trigger PDF indexing."""
        try:
            response = requests.post(f"{self.api_base_url}/index", timeout=300)
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type") == "application/json" else response.text
                return {
                    "error": f"Indexing failed: {error_detail}",
                    "status_code": response.status_code
                }
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def get_docs_count(self) -> Dict[str, Any]:
        """Get count of PDF documents."""
        try:
            response = requests.get(f"{self.api_base_url}/docs-count", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"count": 0, "error": "Failed to get document count"}
        except requests.exceptions.RequestException:
            return {"count": 0, "error": "API unreachable"}
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🔍 PDF Search Engine</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with controls and information."""
        st.sidebar.header("📊 System Status")
        
        # API Health Check
        health_status = self.check_api_health()
        if health_status["status"] == "healthy":
            st.sidebar.success("✅ API is healthy")
            
            # Index statistics
            index_stats = health_status.get("index_stats", {})
            if index_stats:
                st.sidebar.info(f"📚 Indexed: {index_stats.get('total_chunks', 0)} chunks from {index_stats.get('total_documents', 0)} documents")
                st.sidebar.info(f"🧠 Model: {index_stats.get('embedding_model', 'Unknown')}")
                st.sidebar.info(f"💾 Cache: {index_stats.get('cache_size', 0)} entries")
        else:
            st.sidebar.error(f"❌ API Status: {health_status['status']}")
            if "error" in health_status:
                st.sidebar.error(f"Error: {health_status['error']}")
        
        st.sidebar.markdown("---")
        
        # Document Management
        st.sidebar.header("📁 Document Management")
        
        docs_info = self.get_docs_count()
        st.sidebar.info(f"📄 PDF files found: {docs_info.get('count', 0)}")
        
        # Index Management
        if st.sidebar.button("🔄 Reindex Documents"):
            with st.spinner("Indexing documents..."):
                result = self.index_pdfs()
                if "error" in result:
                    st.sidebar.error(f"Indexing failed: {result['error']}")
                else:
                    st.sidebar.success(f"✅ Indexed {result.get('indexed_documents', 0)} documents")
                    st.rerun()
        
        # Clear cache
        if st.sidebar.button("🗑️ Clear Cache"):
            try:
                response = requests.delete(f"{self.api_base_url}/cache")
                if response.status_code == 200:
                    st.sidebar.success("Cache cleared!")
                else:
                    st.sidebar.error("Failed to clear cache")
            except:
                st.sidebar.error("Failed to clear cache")
        
        st.sidebar.markdown("---")
        
        # Search History
        st.sidebar.header("📝 Recent Searches")
        if st.session_state.search_history:
            for i, query in enumerate(reversed(st.session_state.search_history[-10:])):
                if st.sidebar.button(f"🔍 {query[:30]}...", key=f"history_{i}"):
                    st.session_state.search_query = query
                    st.rerun()
        else:
            st.sidebar.info("No recent searches")
        
        # Search Settings
        st.sidebar.header("⚙️ Search Settings")
        max_results = st.sidebar.slider("Max Results", 5, 50, config.MAX_RESULTS)
        return max_results
    
    def render_search_box(self, max_results: int):
        """Render the main search interface."""
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        
        # Search input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Search your PDF documents:",
                placeholder="Enter your search query (e.g., 'machine learning algorithms')",
                key="search_query",
                label_visibility="collapsed"
            )
        
        with col2:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle search
        if (search_clicked or (query and query != st.session_state.last_query)) and query:
            if len(query) < config.MIN_QUERY_LENGTH:
                st.error(f"Query must be at least {config.MIN_QUERY_LENGTH} characters long")
                return
            
            if len(query) > config.MAX_QUERY_LENGTH:
                st.error(f"Query must be no more than {config.MAX_QUERY_LENGTH} characters long")
                return
            
            # Add to search history
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)
            
            st.session_state.last_query = query
            st.session_state.current_page = 1
            
            # Perform search
            with st.spinner("Searching..."):
                start_time = time.time()
                search_results = self.search_pdfs(query, max_results)
                search_time = time.time() - start_time
            
            if "error" in search_results:
                st.error(search_results["error"])
                return
            
            st.session_state.current_results = search_results
            
            # Display search info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Results Found", search_results.get("total_results", 0))
            with col2:
                st.metric("Search Time", f"{search_results.get('search_time', search_time):.3f}s")
            with col3:
                cached = search_results.get("cached", False)
                st.metric("Cache Status", "Hit" if cached else "Miss")
    
    def render_results(self):
        """Render search results with pagination."""
        if not st.session_state.current_results:
            st.info("Enter a search query above to find relevant PDF documents.")
            return
        
        results = st.session_state.current_results.get("results", [])
        
        if not results:
            st.warning("No results found for your query. Try different keywords or check if documents are indexed.")
            return
        
        # Pagination
        total_results = len(results)
        total_pages = (total_results + self.results_per_page - 1) // self.results_per_page
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("← Previous", disabled=st.session_state.current_page <= 1):
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col2:
                st.markdown(f"<div style='text-align: center'>Page {st.session_state.current_page} of {total_pages}</div>", unsafe_allow_html=True)
            
            with col3:
                if st.button("Next →", disabled=st.session_state.current_page >= total_pages):
                    st.session_state.current_page += 1
                    st.rerun()
        
        # Calculate result range for current page
        start_idx = (st.session_state.current_page - 1) * self.results_per_page
        end_idx = min(start_idx + self.results_per_page, total_results)
        page_results = results[start_idx:end_idx]
        
        # Display results
        for i, result in enumerate(page_results, start=start_idx + 1):
            self.render_result_card(i, result)
    
    def render_result_card(self, index: int, result: Dict[str, Any]):
        """Render a single result card."""
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            # Result header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f'<div class="result-header">#{index}: {result["pdf_name"]} (Page {result["page_number"]})</div>', unsafe_allow_html=True)
            with col2:
                confidence = result.get("confidence_score", 0)
                st.markdown(f'<span class="confidence-score">Score: {confidence:.3f}</span>', unsafe_allow_html=True)
            
            # Content preview
            content = result.get("highlight_snippet", result.get("content", ""))
            if len(content) > 500:
                content = content[:500] + "..."
            
            st.markdown(f"**Preview:** {content}")
            
            # Expandable full content
            with st.expander("View Full Content"):
                st.text(result.get("content", ""))
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Vector Score:** {result.get('vector_score', 0):.3f}")
            with col2:
                st.markdown(f"**Keyword Score:** {result.get('keyword_score', 0):.3f}")
            with col3:
                st.markdown(f"**Combined Score:** {result.get('combined_score', 0):.3f}")
            
            # Additional metadata
            metadata = result.get("metadata", {})
            if metadata:
                with st.expander("Metadata"):
                    st.json(metadata)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
    
    def render_analytics(self):
        """Render analytics and insights."""
        if not st.session_state.current_results:
            return
        
        results = st.session_state.current_results.get("results", [])
        if not results:
            return
        
        st.header("📊 Search Analytics")
        
        # Create analytics dataframe
        df_data = []
        for result in results:
            df_data.append({
                "PDF Name": result["pdf_name"],
                "Page": result["page_number"],
                "Confidence": result["confidence_score"],
                "Vector Score": result["vector_score"],
                "Keyword Score": result["keyword_score"],
                "Content Length": len(result["content"])
            })
        
        df = pd.DataFrame(df_data)
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Distribution")
            st.bar_chart(df[["Vector Score", "Keyword Score", "Confidence"]])
        
        with col2:
            st.subheader("Results by PDF")
            pdf_counts = df["PDF Name"].value_counts()
            st.bar_chart(pdf_counts)
        
        # Results table
        st.subheader("Results Summary")
        st.dataframe(df, use_container_width=True)
    
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        
        # Sidebar
        max_results = self.render_sidebar()
        
        # Main content
        tab1, tab2 = st.tabs(["🔍 Search", "📊 Analytics"])
        
        with tab1:
            self.render_search_box(max_results)
            self.render_results()
        
        with tab2:
            self.render_analytics()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.8rem;'>
                PDF Search Engine v1.0.0 | Built with Streamlit, FastAPI, and ChromaDB
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """Main function to run the Streamlit app."""
    try:
        interface = SearchInterface()
        interface.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
