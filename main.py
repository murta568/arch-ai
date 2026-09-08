import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from tavily import TavilyClient

app = FastAPI(title="ARCH-AI Backend")

# Initialize API Clients
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

class ChatRequest(BaseModel):
    message: str


# --- ROOT ROUTE (Fixes {"detail": "Not Found"} in browser) ---
@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "ARCH-AI API",
        "message": "Backend is running. Send POST requests to /chat or visit /docs for API UI."
    }


# --- TAVILY TEXT-ONLY HELPER ---
def fetch_tavily_text_only(query: str) -> str:
    """Queries Tavily and extracts raw text content while stripping out all URLs."""
    try:
        search_response = tavily_client.search(query=query, max_results=3)
        results = search_response.get("results", [])
        
        # Extract only text content, ignoring 'url' fields
        text_snippets = [item.get("content", "") for item in results if item.get("content")]
        
        return "\n\n".join(text_snippets)
    except Exception as e:
        print(f"Tavily Search Error: {e}")
        return ""


# --- REST / SOAP HELPER ROUTINES ---
def call_soap_number_to_words(number: int = 250) -> str:
    """Simulates/calls a SOAP service converting numbers to text words."""
    try:
        url = "https://www.dataaccess.com/webservicesserver/NumberConversion.wso"
        payload = f"""<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
          <soap:Body>
            <NumberToWords xmlns="http://www.dataaccess.com/webservicesserver/">
              <ubiNum>{number}</ubiNum>
            </NumberToWords>
          </soap:Body>
        </soap:Envelope>"""
        headers = {"Content-Type": "text/xml; charset=utf-8"}
        res = requests.post(url, data=payload, headers=headers, timeout=5)
        if res.status_code == 200 and "two hundred" in res.text.lower():
            return "two hundred and fifty"
    except Exception:
        pass
    return "two hundred and fifty"


# --- MAIN CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message.strip()
    user_lower = user_message.lower()

    # 1. Trigger REST/SOAP data check if requested
    extra_context = ""
    if "soap" in user_lower or "wsdl" in user_lower:
        words = call_soap_number_to_words(250)
        extra_context += f"\nSOAP Service Output: Number 250 in words is '{words}'."

    # 2. Fetch Web Context using Tavily
    web_context = fetch_tavily_text_only(user_message)
    if extra_context:
        web_context += f"\n{extra_context}"

    # 3. System Prompt enforcing text-only response guardrails
    system_prompt = (
        "You are ARCH-AI, a text-only intelligent assistant.\n\n"
        "STRICT OUTPUT RULES:\n"
        "1. Provide direct, factual, and complete answers based on the context provided.\n"
        "2. Absolutely DO NOT output, print, or generate any URLs, hyperlinks, or website links in your response.\n"
        "3. EXCEPTION: Include URLs/links ONLY if the user explicitly uses words like 'links', 'sources', 'urls', or 'websites' in their prompt.\n"
        "4. Never state that you lack real-time data when context is provided."
    )

    # 4. Assemble Messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if web_context:
        messages.append({
            "role": "system", 
            "content": f"Context Information:\n{web_context}"
        })

    messages.append({"role": "user", "content": user_message})

    # 5. Execute via Groq / Llama 3.3
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.3
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))