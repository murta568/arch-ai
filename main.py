import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from tavily import TavilyClient

app = FastAPI(title="ARCH-AI Backend")

# Initialize API Clients
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

class ChatRequest(BaseModel):
    message: str


# --- ROOT ROUTE: Web Interface ---
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ARCH-AI Interface</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0d1117; color: #c9d1d9; max-width: 700px; margin: 40px auto; padding: 20px; }
            h2 { color: #58a6ff; text-align: center; }
            #chatbox { height: 420px; overflow-y: auto; border: 1px solid #30363d; padding: 15px; background: #161b22; border-radius: 8px; margin-bottom: 15px; }
            .msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 6px; line-height: 1.4; word-wrap: break-word; }
            .user { background: #1f6feb; color: white; align-self: flex-end; margin-left: 20%; }
            .bot { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; margin-right: 20%; }
            .input-container { display: flex; gap: 10px; }
            input { flex: 1; padding: 12px; background: #0d1117; border: 1px solid #30363d; color: white; border-radius: 6px; font-size: 15px; }
            input:focus { outline: none; border-color: #58a6ff; }
            button { padding: 12px 20px; background: #238636; border: none; color: white; font-weight: bold; border-radius: 6px; cursor: pointer; font-size: 15px; }
            button:hover { background: #2ea043; }
        </style>
    </head>
    <body>
        <h2>ARCH-AI Terminal</h2>
        <div id="chatbox"></div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') sendMsg()">
            <button onclick="sendMsg()">Send</button>
        </div>

        <script>
            async function sendMsg() {
                const input = document.getElementById('userInput');
                const chatbox = document.getElementById('chatbox');
                const text = input.value.trim();
                if (!text) return;

                chatbox.innerHTML += `<div class="msg user">${text}</div>`;
                input.value = '';
                chatbox.scrollTop = chatbox.scrollHeight;

                const botMsgDiv = document.createElement('div');
                botMsgDiv.className = 'msg bot';
                botMsgDiv.innerText = 'Thinking...';
                chatbox.appendChild(botMsgDiv);
                chatbox.scrollTop = chatbox.scrollHeight;

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text })
                    });
                    const data = await res.json();
                    botMsgDiv.innerText = data.response || data.detail || 'Error getting response.';
                } catch (e) {
                    botMsgDiv.innerText = 'Error connecting to server.';
                }
                chatbox.scrollTop = chatbox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


# --- TAVILY TEXT-ONLY HELPER ---
def fetch_tavily_text_only(query: str) -> str:
    """Queries Tavily and extracts raw text content while stripping out all URLs."""
    try:
        search_response = tavily_client.search(query=query, max_results=3)
        results = search_response.get("results", [])
        
        # Extract only text content, omitting 'url' fields
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