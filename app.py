import streamlit as st
import requests
import json
from typing import Dict, List, Any, TypedDict, Annotated
import os
from dataclasses import dataclass
from datetime import datetime
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator
from typing import Optional
import time
import random

# Configuration
@dataclass
class Config:
    TAVILY_API_KEY: str = ""
    LM_STUDIO_URL: str = "http://127.0.0.1:1234/v1"
    LM_STUDIO_MODEL: str = "google/gemma-3-1b"
    REQUEST_TIMEOUT: int = 180
    MAX_RETRIES: int = 3

# State definition for LangGraph
class ResearchState(TypedDict):
    topic: str
    research_plan: str
    search_results: str
    analysis: str
    final_report: str
    messages: Annotated[List[Any], operator.add]
    next_step: str
    error: Optional[str]

class LMStudioLLM(LLM):
    """Custom LLM wrapper for LM Studio with improved error handling"""
    
    base_url: str = "http://localhost:1234/v1"
    model: str = "microsoft/Phi-4-mini-reasoning"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 180
    max_retries: int = 3
    
    @property
    def _llm_type(self) -> str:
        return "lm_studio"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LM Studio API with retry logic"""
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                session = requests.Session()
                
                response = session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    if attempt == self.max_retries - 1:
                        return f"Error: {response.status_code} - {response.text}"
                    continue
                    
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    return f"Error: Request timed out after {self.timeout} seconds. Please check if LM Studio is running and responsive."
            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries - 1:
                    return f"Error: Cannot connect to LM Studio at {self.base_url}. Please ensure LM Studio is running."
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    return f"Error connecting to LM Studio: {str(e)}"
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        return "Error: All retry attempts failed"

class TavilySearchTool(BaseTool):
    """Tavily AI-powered search tool"""
    
    name: str = "tavily_search"
    description: str = "Search the web using Tavily AI search API for current information"
    api_key: str = ""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute search using Tavily API"""
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": True,
            "max_results": max_results,
            "include_domains": [],
            "exclude_domains": []
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Create a clean, professional report format
                formatted_output = f"""
## Search Results for: "{query}"

### Summary
{data.get("answer", "No summary available")}

### Sources
"""
                
                sources = data.get("results", [])
                if sources:
                    for i, source in enumerate(sources, 1):
                        formatted_output += f"""
**{i}. {source.get("title", "Untitled Source")}**
- **URL:** {source.get("url", "#")}
- **Content:** {source.get("content", "No content available")[:300]}...

---
"""
                
                return formatted_output
            else:
                return f"Tavily API Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Search Error: {str(e)}"

class ResearchGraph:
    """LangGraph-based research orchestration system"""
    
    def __init__(self, tavily_api_key: str, lm_studio_url: str, model_name: str):
        self.llm = LMStudioLLM(base_url=lm_studio_url, model=model_name, timeout=180)
        self.search_tool = TavilySearchTool(api_key=tavily_api_key)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        workflow.add_node("research_planner", self._research_planner_node)
        workflow.add_node("web_searcher", self._web_searcher_node)
        
        workflow.set_entry_point("research_planner")
        workflow.add_edge("research_planner", "web_searcher")
        workflow.add_edge("web_searcher", END)
        
        return workflow.compile()
    
    def _research_planner_node(self, state: ResearchState) -> ResearchState:
        """Research planning node"""
        
        prompt = f"""
        As a research methodology expert, create a detailed research plan to investigate this topic: "{state['topic']}"

        Your plan should include:
        1. Key questions that need to be answered
        2. Types of sources to prioritize (academic, news, official, etc.)
        3. Multiple search strategies and keyword variations
        4. Red flags or potential biases to watch for
        5. Verification methods and fact-checking approaches

        Make the plan specific, actionable, and thorough for this particular topic.
        """
        
        try:
            research_plan = self.llm(prompt)
            
            if "Error:" in research_plan or "timed out" in research_plan.lower():
                research_plan = f"""
## Research Plan for: {state['topic']}

### Key Questions
- What is the current status of this topic?
- What evidence supports or contradicts this claim?
- What do authoritative sources say?

### Search Strategy
- Direct topic search
- Evidence-based search
- Expert opinion search
- Recent news search

### Source Prioritization
- Official sources first
- Peer-reviewed content
- Reputable news outlets
- Expert commentary

### Verification Methods
- Cross-reference multiple sources
- Check publication dates
- Verify author credentials
- Look for consensus
                """
            
            state["research_plan"] = research_plan
            state["messages"].append(AIMessage(content=f"Research plan created for: {state['topic']}"))
            state["next_step"] = "web_searcher"
        except Exception as e:
            state["error"] = f"Error in research planning: {str(e)}"
            state["research_plan"] = f"Basic research plan for: {state['topic']} (Error occurred in detailed planning)"
        
        return state
    
    def _web_searcher_node(self, state: ResearchState) -> ResearchState:
        """Web searching node with Tavily"""
        
        topic = state["topic"]
        search_queries = [
            topic,
            f"{topic} facts evidence 2024",
            f"{topic} expert analysis",
            f"{topic} recent research findings"
        ]
        
        all_results = []
        successful_searches = 0
        
        try:
            for i, query in enumerate(search_queries):
                try:
                    result = self.search_tool._run(query, max_results=3)
                    if "Error:" not in result:
                        successful_searches += 1
                    all_results.append(f"### Search Query {i+1}: '{query}'\n{result}\n")
                except Exception as e:
                    all_results.append(f"**Search {i+1}** for '{query}' failed: {str(e)}")
            
            combined_results = "\n".join(all_results)
            state["search_results"] = combined_results
            state["messages"].append(AIMessage(content=f"Completed {len(search_queries)} searches ({successful_searches} successful) for: {topic}"))
            state["next_step"] = "complete"
            
        except Exception as e:
            state["error"] = f"Error in web searching: {str(e)}"
            state["search_results"] = f"Error occurred during search phase for: {topic}"
        
        return state
    
    def research_topic(self, topic: str) -> Dict[str, str]:
        """Execute the research workflow"""
        
        initial_state = ResearchState(
            topic=topic,
            research_plan="",
            search_results="",
            analysis="",
            final_report="",
            messages=[HumanMessage(content=f"Research topic: {topic}")],
            next_step="research_planner",
            error=None
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "research_plan": final_state.get("research_plan", "Not available"),
                "search_results": final_state.get("search_results", "Not available"),
                "error": final_state.get("error")
            }
        except Exception as e:
            return {
                "research_plan": "Error occurred during workflow execution",
                "search_results": "Error occurred during workflow execution",
                "error": f"Graph execution error: {str(e)}"
            }

def test_connections(tavily_key: str, lm_studio_url: str, model_name: str) -> Dict[str, bool]:
    """Test API connections"""
    results = {"tavily": False, "lm_studio": False}
    
    # Test Tavily API
    if tavily_key:
        try:
            search_tool = TavilySearchTool(api_key=tavily_key)
            test_result = search_tool._run("hello world test", max_results=1)
            results["tavily"] = "Error:" not in test_result
        except:
            results["tavily"] = False
    
    # Test LM Studio
    try:
        test_llm = LMStudioLLM(base_url=lm_studio_url, model=model_name, timeout=30)
        test_response = test_llm("Say 'Hello'")
        results["lm_studio"] = "Error:" not in test_response and "timed out" not in test_response.lower()
    except:
        results["lm_studio"] = False
    
    return results

def main():
    # Page configuration
    st.set_page_config(
        page_title="Research Intelligence Hub",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Clean, professional CSS
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #fafafa;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Status indicators */
    .status-success {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-section h3 {
        color: #374151;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        margin-top: 0;
    }
    
    /* Progress indicator */
    .progress-step {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: #f3f4f6;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .progress-step.active {
        background: #ede9fe;
        border-left-color: #8b5cf6;
    }
    
    .progress-step.complete {
        background: #d1fae5;
        border-left-color: #10b981;
    }
    
    /* Clean button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Research Intelligence Hub</h1>
        <p>AI-Powered Research & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3>🔑 API Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            placeholder="Enter your Tavily API key...",
            help="Get your API key from tavily.com"
        )
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>🤖 Model Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        lm_studio_url = st.text_input(
            "LM Studio URL",
            value="http://localhost:1234/v1",
            help="Local LM Studio endpoint"
        )
        
        model_name = st.text_input(
            "Model Name",
            value="google/gemma-3-1b",
            help="Model identifier in LM Studio"
        )
        
        # Connection status
        if st.button("Test Connections", use_container_width=True):
            with st.spinner("Testing connections..."):
                status = test_connections(tavily_key, lm_studio_url, model_name)
                
                if status["tavily"]:
                    st.markdown('<div class="status-success">✅ Tavily API Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">❌ Tavily API Failed</div>', unsafe_allow_html=True)
                
                if status["lm_studio"]:
                    st.markdown('<div class="status-success">✅ LM Studio Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">❌ LM Studio Failed</div>', unsafe_allow_html=True)
    
    # Main content area
    col1 = st.columns(1)[0]
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2 style="margin-top: 0; color: #374151;">Research Query</h2>
        </div>
        """, unsafe_allow_html=True)
        
        topic = st.text_area(
            "Enter your research topic or question:",
            placeholder="Example: What are the latest developments in quantum computing applications for cryptography?",
            height=120,
            help="Be specific and detailed for better research results"
        )
        
        if st.button("🚀 Start Research", use_container_width=True, type="primary"):
            if not topic.strip():
                st.error("Please enter a research topic.")
            elif not tavily_key:
                st.error("Please provide a Tavily API key in the sidebar.")
            else:
                # Research execution
                with st.spinner("Conducting research..."):
                    try:
                        research_graph = ResearchGraph(tavily_key, lm_studio_url, model_name)
                        results = research_graph.research_topic(topic)
                        
                        # Store results in session state
                        st.session_state.research_results = results
                        st.session_state.research_topic = topic
                        
                        st.success("Research completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Research failed: {str(e)}")
    
    # Display results if available
    if hasattr(st.session_state, 'research_results') and st.session_state.research_results:
        st.markdown("---")
        
        # Results header
        st.markdown(f"""
        <div class="card">
            <h2 style="margin-top: 0; color: #374151;">Research Results: {st.session_state.research_topic}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Results tabs
        tab1, tab2 = st.tabs(["📋 Research Plan", "🔍 Search Results"])
        
        with tab1:
            st.markdown("""
            <div class="card">
            """, unsafe_allow_html=True)
            
            if st.session_state.research_results.get("research_plan"):
                st.markdown(st.session_state.research_results["research_plan"])
            else:
                st.warning("Research plan not available")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="card">
            """, unsafe_allow_html=True)
            
            if st.session_state.research_results.get("search_results"):
                st.markdown(st.session_state.research_results["search_results"])
            else:
                st.warning("Search results not available")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show errors if any
        if st.session_state.research_results.get("error"):
            st.markdown(f"""
            <div class="status-warning">
                ⚠️ {st.session_state.research_results["error"]}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
