import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from tavily import TavilyClient

app = FastAPI()

# Initialize clients (Ensure API keys are set in your environment)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

class ChatRequest(BaseModel):
    message: str

def fetch_tavily_text_only(query: str) -> str:
    """Queries Tavily and extracts raw text content while stripping out URLs."""
    try:
        search_response = tavily_client.search(query=query, max_results=3)
        results = search_response.get("results", [])
        
        # Extract only the text snippet, omitting 'url' fields completely
        text_snippets = [item.get("content", "") for item in results if item.get("content")]
        
        return "\n\n".join(text_snippets)
    except Exception as e:
        print(f"Tavily Search Error: {e}")
        return ""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message.strip()
    
    # 1. Fetch live web context using Tavily
    web_context = fetch_tavily_text_only(user_message)
    
    # 2. Strict system prompt preventing link insertion unless explicitly requested
    system_prompt = (
        "You are ARCH-AI, an intelligent assistant. "
        "Use the provided Web Search Context to answer the user's question directly, accurately, and completely.\n\n"
        "STRICT OUTPUT RULES:\n"
        "1. Answer the query directly using the factual information from the context.\n"
        "2. DO NOT output, print, or generate any URLs, hyperlinks, or website links in your response.\n"
        "3. EXCEPTION: Include links ONLY if the user explicitly uses words like 'links', 'sources', 'urls', or 'websites' in their prompt.\n"
        "4. Never state that you lack real-time or live data when context is provided above."
    )
    
    # 3. Construct message payload
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    if web_context:
        messages.append({
            "role": "system", 
            "content": f"Web Search Context (Information Only):\n{web_context}"
        })
        
    messages.append({"role": "user", "content": user_message})
    
    # 4. Generate response via Groq
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.3
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))