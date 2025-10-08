"""
Memory Agent for thread-aware query caching.

This agent checks if similar queries have been asked in the current conversation
thread and returns cached answers when appropriate, reducing redundant processing.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from models import GraphState, AgentError, RetrieverError, AGENT_HANDOFFS
from decorators import agent_error_handler
from utils import validate_state


# Initialize query cache with vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_cache_store = Chroma(
    persist_directory="query_cache_vectorstore",
    embedding_function=embedding_model,
    collection_name="query_cache"
)

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Memory"]


@agent_error_handler
def memory_agent(state: GraphState):
    """
    Check for cached answers in the current conversation thread.
    
    This agent searches for similar queries within the same thread and returns
    cached answers if a sufficiently similar query (>90% similarity) is found.
    
    Args:
        state: Current graph state containing query and thread information
        
    Returns:
        dict: Updated state with cache hit status and next agent routing
        
    Raises:
        AgentError: If state validation fails
        ValueError: If query is empty
    """
    print("---NODE: Memory Agent---")
    
    if not validate_state(state):
        raise AgentError("Invalid state received")
    
    query = state["query"]
    thread_id = state.get("thread_id", "default")
    
    print(f"Thread ID: {thread_id}")
    
    if not query or len(query.strip()) == 0:
        raise ValueError("Empty query received")
    
    # Check if cache bypass is requested
    if state.get("bypass_cache", False):
        print("Cache bypass flag set")
        
        state["trace"].append({
            "agent": "Memory",
            "tool": "Cache bypass",
            "output": "Cache bypassed",
            "handoff-to": "Retriever",
            "reasoning": "New conversation started"
        })
        
        return {
            "next_agent": "Retriever",
            "cache_hit": False
        }
    
    try:
        # Search for similar queries within this thread
        similar_queries = query_cache_store.similarity_search_with_score(
            query, 
            k=5,
            filter={"thread_id": thread_id}
        )
        
        if not similar_queries or len(similar_queries) == 0:
            print("No cached queries found in this thread")
            
            state["trace"].append({
                "agent": "Memory",
                "tool": "Query Cache",
                "output": "No cache hit in thread",
                "handoff-to": "Retriever",
                "reasoning": f"New query in thread {thread_id}"
            })
            
            return {
                "next_agent": "Retriever",
                "cache_hit": False
            }
        
        # Get best matching query
        best_match, distance = similar_queries[0]
        
        if not best_match or not hasattr(best_match, 'page_content'):
            raise RetrieverError("Invalid cache data")
        
        # Calculate similarity score (inverse of distance)
        similarity = 1.0 / (1.0 + distance)
        print(f"Best match similarity: {similarity:.3f} (thread: {thread_id})")
        
        # Cache hit threshold: 90% similarity
        if similarity >= 0.90:
            print(f"Cache hit in thread {thread_id}")
            
            cached_answer = best_match.metadata.get("answer", "")
            
            if not cached_answer:
                return {
                    "next_agent": "Retriever",
                    "cache_hit": False
                }
            
            state["trace"].append({
                "agent": "Memory",
                "tool": "Query Cache",
                "output": f"Cache hit in thread {thread_id}",
                "handoff-to": "Aggregator",
                "reasoning": "Identical query found in this conversation"
            })
            
            return {
                "cache_hit": True,
                "cached_answer": cached_answer,
                "next_agent": "Aggregator"
            }
        
        else:
            print(f"New query in thread {thread_id} (similarity: {similarity:.3f})")
            
            state["trace"].append({
                "agent": "Memory",
                "tool": "Query Cache",
                "output": f"Low similarity ({similarity:.3f})",
                "handoff-to": "Retriever",
                "reasoning": "Different query"
            })
            
            return {
                "next_agent": "Retriever",
                "cache_hit": False
            }
    
    except Exception as e:
        print(f"Cache error: {e}")
        
        state["trace"].append({
            "agent": "Memory",
            "tool": "Query Cache",
            "error": str(e),
            "handoff-to": "Retriever",
            "reasoning": "Cache error"
        })
        
        return {
            "next_agent": "Retriever",
            "cache_hit": False
        }
