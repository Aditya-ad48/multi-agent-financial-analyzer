"""
Aggregator Agent for final answer generation.

This agent synthesizes information from all previous agents, generates
the final response, and caches results for future queries in the same thread.
"""

import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from models import GraphState, AGENT_HANDOFFS
from decorators import agent_error_handler
from config import GROQ_API_KEY


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)

# Initialize cache store for thread-aware query caching
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_cache_store = Chroma(
    persist_directory="query_cache_vectorstore",
    embedding_function=embedding_model,
    collection_name="query_cache"
)

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Aggregator"]


@agent_error_handler
def aggregator_agent(state: GraphState):
    """
    Generate final answer from all available information.
    
    This agent synthesizes data from document retrieval, web search, table extraction,
    and calculations to produce a comprehensive answer. It handles casual queries,
    conversation history requests, and error cases, then caches results for future use.
    
    Args:
        state: Current graph state containing query, context, and conversation data
        
    Returns:
        dict: Final answer and END routing
    """
    print("---NODE: Aggregator---")
    
    context = state.get("context", {})
    user_query = state['query']
    query_lower = user_query.lower()
    thread_id = state.get("thread_id", "default")
    
    print(f"Processing in thread: {thread_id}")
    
    # Return cached answer if available
    if state.get("cache_hit") and state.get("cached_answer"):
        return {"answer": state['cached_answer'], "next_agent": "END"}
    
    # Handle casual queries (greetings, thanks, compliments)
    if context.get("casual_query"):
        query_lower = user_query.lower()
        
        if "thank" in query_lower:
            answer = "You're welcome! Let me know if you have any questions about financial data or documents."
        elif "hello" in query_lower or "hi" in query_lower or "hey" in query_lower:
            answer = "Hello! I'm here to help with financial document analysis and queries. What would you like to know?"
        elif "good" in query_lower or "great" in query_lower or "awesome" in query_lower:
            answer = "Thank you! I'm designed to assist with financial analysis and queries. Feel free to ask me anything about documents, data, or calculations."
        else:
            answer = "I'm here to help with financial document analysis. What would you like to know?"
        
        state["trace"].append({
            "agent": "Aggregator",
            "tool": "Casual response",
            "output": "Polite response",
            "handoff-to": "END",
            "reasoning": "Casual query handled"
        })
        
        return {"answer": answer, "next_agent": "END"}
    
    # Handle cases where no usable data was found
    if context.get("no_usable_data"):
        checked_docs = "Yes" if context.get("retrieved_docs") else "No"
        checked_web = "Yes" if context.get("web_results") else "No"
        
        error_answer = f"""I couldn't find relevant information to answer: "{user_query}"

**What I tried:**
- Searched local documents: {checked_docs}
- Searched the web: {checked_web}

The available information doesn't address your question. Please try:
- Rephrasing your query
- Uploading relevant documents
- Asking a different question"""
        
        state["trace"].append({
            "agent": "Aggregator",
            "tool": "Error Handler",
            "output": "No usable data",
            "handoff-to": "END",
            "reasoning": "All data sources insufficient"
        })
        
        return {"answer": error_answer, "next_agent": "END"}
    
    # Handle conversation history queries
    conversation_keywords = [
        "previous query", "what did i ask", "earlier", "before", "last question",
        "all queries", "conversation", "queries i asked", "brief of", "history"
    ]
    asks_about_conversation = any(phrase in query_lower for phrase in conversation_keywords)
    
    if asks_about_conversation:
        conversation_history = state.get("conversation_history", [])
        
        if not conversation_history:
            answer = "This is the first query in this conversation."
        else:
            history_text = "\n".join(conversation_history[-10:])
            
            prompt = f"""The user is asking about their conversation history.

CONVERSATION HISTORY:
{history_text}

USER QUERY: {user_query}

Provide a brief summary of what was asked and answered."""
            
            try:
                response = llm.invoke(prompt)
                answer = response.content
            except:
                answer = f"Conversation history:\n\n{history_text}"
        
        return {"answer": answer, "next_agent": "END"}
    
    # Check for available data to answer the query
    has_data = (
        context.get("retrieved_docs") or 
        context.get("web_results") or 
        context.get("structured_data") or
        context.get("calculations")
    )
    
    if not has_data:
        error_answer = f"""Unable to retrieve information to answer: "{user_query}"

Please try rephrasing or ensure documents are uploaded."""
        return {"answer": error_answer, "next_agent": "END"}
    
    # Build comprehensive context from all available sources
    context_summary = []
    
    if context.get("retrieved_docs"):
        docs_preview = "\n".join(context["retrieved_docs"][:3])[:3000]
        context_summary.append(f"**Documents:**\n{docs_preview}")
    
    if context.get("web_results"):
        web_preview = "\n".join(context["web_results"][:2])[:2000]
        context_summary.append(f"**Web Results:**\n{web_preview}")
    
    if context.get("structured_data"):
        data_str = json.dumps(context["structured_data"], indent=2)[:2000]
        context_summary.append(f"**Data:**\n{data_str}")
    
    if context.get("calculations"):
        calc_str = json.dumps(context["calculations"], indent=2)[:1000]
        context_summary.append(f"**Calculations:**\n{calc_str}")
    
    context_text = "\n\n".join(context_summary) if context_summary else "(No context)"
    
    # Generate final answer using LLM
    prompt = f"""You are an expert financial analyst assistant providing clear, accurate responses.

===== USER'S QUESTION =====
{user_query}

===== AVAILABLE INFORMATION =====
{context_text}

===== INSTRUCTIONS =====
1. Start with a direct answer (1-2 sentences)
2. Use specific data points, numbers, and facts from the context
3. Structure your response clearly with headers if needed
4. Be comprehensive but concise
5. Use **bold** for key numbers
6. Do NOT mention "based on the context" - just answer directly

===== YOUR ANSWER =====
"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content
        
        # Cache the answer for future queries in this thread
        try:
            query_cache_store.add_texts(
                texts=[state["query"]],
                metadatas=[{
                    "answer": answer, 
                    "timestamp": datetime.now().isoformat(),
                    "thread_id": thread_id
                }]
            )
            print(f"Answer cached in thread: {thread_id}")
        except Exception as cache_error:
            print(f"Cache save failed: {cache_error}")
        
        state["trace"].append({
            "agent": "Aggregator",
            "tool": "LLM",
            "output": "Answer generated",
            "handoff-to": "END",
            "reasoning": "Answer synthesized"
        })
        
        return {"answer": answer, "next_agent": "END"}
    
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return {"answer": f"Error generating response for: {user_query}", "next_agent": "END"}
