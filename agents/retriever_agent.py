"""
Retriever Agent for document retrieval from vector store.

This agent retrieves relevant documents from the vector database
and forwards them to the Validator for assessment and routing.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from models import GraphState, RetrieverError, AGENT_HANDOFFS
from decorators import agent_error_handler, retry_with_exponential_backoff
from config import DEFAULT_DB_PATH


# Initialize vector store and retriever
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=DEFAULT_DB_PATH, 
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Retriever"]


@agent_error_handler
@retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
def retrieve_agent(state: GraphState):
    """
    Retrieve relevant documents from vector store.
    
    This agent queries the vector database for documents relevant to the user's
    query and forwards them to the Validator for relevance assessment and routing.
    
    Args:
        state: Current graph state containing the query
        
    Returns:
        dict: Updated state with retrieved documents and next agent routing
        
    Raises:
        ValueError: If query is empty
        RetrieverError: If no valid documents are retrieved
    """
    print("---NODE: Retriever---")
    print("Retrieving documents from vector store")
    
    query = state["query"]
    
    if not query:
        raise ValueError("Empty query for retrieval")
    
    try:
        # Retrieve documents from vector store
        docs = retriever.invoke(query)
        
        if not docs:
            print("No documents found in vector store")
            raise RetrieverError("Vector store returned no documents")
        
        # Extract and validate document content
        context = [doc.page_content for doc in docs]
        valid_context = [c for c in context if c and len(c.strip()) > 10]
        
        if not valid_context:
            print("Retrieved documents contain no useful content")
            raise RetrieverError("Retrieved documents have no useful content")
        
        print(f"Retrieved {len(valid_context)} valid documents")
        print("Forwarding to Validator for assessment")
        
        state["trace"].append({
            "agent": "Retriever",
            "tool": "vector_store",
            "input": query,
            "output": f"Retrieved {len(valid_context)} documents",
            "handoff-to": "Validator",
            "reasoning": "Documents retrieved successfully, sending to Validator for assessment"
        })
        
        return {
            "context": {
                **state.get("context", {}), 
                "retrieved_docs": valid_context,
                "data_source": "documents"
            },
            "next_agent": "Validator"
        }
            
    except Exception as e:
        # If retrieval fails, fallback to WebSearch
        print(f"Retrieval failed: {e}")
        print("Routing to WebSearch for external data")
        
        state["trace"].append({
            "agent": "Retriever",
            "tool": "vector_store",
            "error": str(e),
            "output": "Document retrieval failed",
            "handoff-to": "WebSearch",
            "reasoning": "Vector store retrieval failed, trying WebSearch instead"
        })
        
        return {
            "context": {
                **state.get("context", {}), 
                "retriever_failed": True
            },
            "next_agent": "WebSearch"
        }
