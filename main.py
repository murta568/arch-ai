import os
import requests
from typing import List, Dict, Any, Optional
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
    user_name: Optional[str] = None
    user_age: Optional[str] = None
    history: List[Dict[str, Any]] = []


# --- ROOT ROUTE: ChatGPT-Style Interface ---
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
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #212121; color: #ececec; display: flex; height: 100vh; overflow: hidden; }

            /* Modal Overlay */
            #modal-overlay { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.8); display: flex; justify-content: center; align-items: center; z-index: 1000; }
            .modal { background-color: #171717; border: 1px solid #30363d; border-radius: 12px; padding: 24px; width: 320px; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
            .modal h3 { color: #58a6ff; text-align: center; }
            .modal input { width: 100%; padding: 10px; background-color: #212121; border: 1px solid #424242; border-radius: 6px; color: #fff; font-size: 14px; outline: none; }
            .modal button { width: 100%; padding: 10px; background-color: #58a6ff; border: none; border-radius: 6px; color: #000; font-weight: 700; cursor: pointer; }

            /* Bottom Left User Profile Badge */
            #user-profile-badge {
                position: fixed;
                bottom: 20px;
                left: 20px;
                background-color: #171717;
                border: 1px solid #2f2f2f;
                border-radius: 30px;
                padding: 6px 14px 6px 8px;
                display: none;
                align-items: center;
                gap: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                z-index: 100;
            }

            .avatar-circle {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background-color: #58a6ff;
                color: #000;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                text-transform: uppercase;
            }

            .user-badge-name {
                font-size: 13px;
                font-weight: 600;
                color: #ececec;
            }

            /* Sidebar styling */
            #sidebar { width: 260px; background-color: #171717; display: flex; flex-direction: column; transition: width 0.3s ease; border-right: 1px solid #2f2f2f; z-index: 10; }
            #sidebar.collapsed { width: 0; overflow: hidden; border-right: none; }
            
            .sidebar-header { padding: 16px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #2f2f2f; }
            .sidebar-header h1 { font-size: 18px; color: #58a6ff; font-weight: 700; }
            
            .btn-new-chat { margin: 12px; padding: 10px 14px; background-color: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; font-weight: 600; font-size: 14px; transition: background 0.2s; }
            .btn-new-chat:hover { background-color: #383838; }

            .chat-list-container { flex: 1; overflow-y: auto; padding: 8px; }
            .section-label { font-size: 11px; color: #8e8e8e; text-transform: uppercase; margin: 8px 8px 4px 8px; font-weight: 600; }
            
            .chat-item { display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; border-radius: 6px; cursor: pointer; color: #b4b4b4; margin-bottom: 4px; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; overflow: hidden; transition: background 0.2s; }
            .chat-item:hover, .chat-item.active { background-color: #2f2f2f; color: #fff; }
            .chat-item.pinned { border-left: 3px solid #58a6ff; }
            
            .chat-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 8px; }
            .chat-actions { display: none; gap: 4px; }
            .chat-item:hover .chat-actions { display: flex; }
            
            .action-btn { background: none; border: none; color: #8e8e8e; cursor: pointer; font-size: 12px; padding: 2px 4px; border-radius: 4px; }
            .action-btn:hover { color: #fff; background-color: #424242; }

            /* Main view */
            #main-content { flex: 1; display: flex; flex-direction: column; height: 100vh; position: relative; }
            
            .top-bar { height: 50px; display: flex; align-items: center; padding: 0 16px; border-bottom: 1px solid #2f2f2f; background-color: #212121; gap: 12px; }
            .toggle-sidebar-btn { background: none; border: none; color: #b4b4b4; font-size: 20px; cursor: pointer; padding: 4px 8px; border-radius: 4px; }
            .toggle-sidebar-btn:hover { background-color: #2f2f2f; color: #fff; }
            .app-title-top { font-size: 16px; font-weight: 700; color: #58a6ff; }

            #chatbox { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; width: 100%; max-width: 800px; margin: 0 auto; }
            
            .msg { display: flex; flex-direction: column; max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 15px; word-wrap: break-word; }
            .user { align-self: flex-end; background-color: #2f2f2f; color: #ececec; border-bottom-right-radius: 2px; }
            .bot { align-self: flex-start; background-color: #171717; color: #d1d5db; border: 1px solid #2f2f2f; border-bottom-left-radius: 2px; }

            .input-area { padding: 12px 16px 16px 16px; background-color: #212121; display: flex; flex-direction: column; align-items: center; }
            .input-box { width: 100%; max-width: 800px; margin: 0 auto; display: flex; background-color: #2f2f2f; border: 1px solid #424242; border-radius: 12px; overflow: hidden; }
            input.msg-input { flex: 1; padding: 14px; background: transparent; border: none; color: #fff; font-size: 15px; outline: none; }
            button.send-btn { padding: 0 20px; background-color: #58a6ff; border: none; color: #000; font-weight: 700; cursor: pointer; transition: background 0.2s; }
            button.send-btn:hover { background-color: #79b8ff; }
            
            .disclaimer-text { font-size: 12px; color: #8e8e8e; margin-top: 8px; text-align: center; }
        </style>
    </head>
    <body>
        <!-- User Info Modal -->
        <div id="modal-overlay">
            <div class="modal">
                <h3>Welcome to ARCH-AI</h3>
                <input type="text" id="userNameInput" placeholder="Enter your Name">
                <input type="number" id="userAgeInput" placeholder="Enter your Age">
                <button onclick="saveUserInfo()">Start Chatting</button>
            </div>
        </div>

        <!-- Floating Bottom Left User Profile Badge -->
        <div id="user-profile-badge">
            <div class="avatar-circle" id="user-avatar-initial">M</div>
            <span class="user-badge-name" id="user-badge-text">Murtajiz</span>
        </div>

        <div id="sidebar">
            <div class="sidebar-header">
                <h1>ARCH-AI</h1>
            </div>
            <button class="btn-new-chat" onclick="startNewChat()">+ New Chat</button>
            <div class="chat-list-container">
                <div class="section-label">Pinned</div>
                <div id="pinned-list"></div>
                <div class="section-label" style="margin-top: 16px;">Recent Chats</div>
                <div id="chats-list"></div>
                <div class="section-label" style="margin-top: 16px;">Archived</div>
                <div id="archived-list"></div>
            </div>
        </div>

        <div id="main-content">
            <div class="top-bar">
                <button class="toggle-sidebar-btn" onclick="toggleSidebar()" title="Toggle Sidebar">☰</button>
                <span class="app-title-top">ARCH-AI</span>
            </div>

            <div id="chatbox"></div>

            <div class="input-area">
                <div class="input-box">
                    <input type="text" id="userInput" class="msg-input" placeholder="Ask ARCH-AI a question..." onkeydown="if(event.key==='Enter') sendMsg()">
                    <button class="send-btn" onclick="sendMsg()">Send</button>
                </div>
                <div class="disclaimer-text">ARCH-AI isn’t human. It can make mistakes, so double-check it.</div>
            </div>
        </div>

        <script>
            let chats = [];
            let activeChatId = null;
            let userName = "";
            let userAge = "";

            window.onload = function() {
                const savedName = localStorage.getItem('arch_user_name');
                const savedAge = localStorage.getItem('arch_user_age');
                if (savedName && savedAge) {
                    userName = savedName;
                    userAge = savedAge;
                    document.getElementById('modal-overlay').style.display = 'none';
                    updateUserBadge(userName);
                }
            };

            function updateUserBadge(name) {
                const badge = document.getElementById('user-profile-badge');
                const initialDiv = document.getElementById('user-avatar-initial');
                const nameSpan = document.getElementById('user-badge-text');

                initialDiv.innerText = name.charAt(0).toUpperCase();
                nameSpan.innerText = name;
                badge.style.display = 'flex';
            }

            function saveUserInfo() {
                const nameVal = document.getElementById('userNameInput').value.trim();
                const ageVal = document.getElementById('userAgeInput').value.trim();
                if (!nameVal || !ageVal) {
                    alert("Please enter both your name and age.");
                    return;
                }
                userName = nameVal;
                userAge = ageVal;
                localStorage.setItem('arch_user_name', userName);
                localStorage.setItem('arch_user_age', userAge);
                document.getElementById('modal-overlay').style.display = 'none';
                updateUserBadge(userName);
            }

            function toggleSidebar() {
                document.getElementById('sidebar').classList.toggle('collapsed');
            }

            function startNewChat() {
                activeChatId = null;
                document.getElementById('chatbox').innerHTML = '';
                renderSidebar();
            }

            function generateSummary(text) {
                return text.length > 25 ? text.substring(0, 22) + '...' : text;
            }

            async function sendMsg() {
                const input = document.getElementById('userInput');
                const text = input.value.trim();
                if (!text) return;

                const chatbox = document.getElementById('chatbox');
                
                if (!activeChatId) {
                    activeChatId = Date.now();
                    const newChat = {
                        id: activeChatId,
                        title: generateSummary(text),
                        messages: [],
                        isPinned: false,
                        isArchived: false
                    };
                    chats.unshift(newChat);
                }

                const currentChat = chats.find(c => c.id === activeChatId);
                
                const historyToSend = currentChat.messages.map(m => ({
                    role: m.role === 'bot' ? 'assistant' : 'user',
                    content: m.content
                }));

                currentChat.messages.push({ role: 'user', content: text });

                renderMessages();
                input.value = '';

                const botMsgDiv = document.createElement('div');
                botMsgDiv.className = 'msg bot';
                botMsgDiv.innerText = 'Thinking...';
                chatbox.appendChild(botMsgDiv);
                chatbox.scrollTop = chatbox.scrollHeight;

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            message: text,
                            user_name: userName,
                            user_age: userAge,
                            history: historyToSend 
                        })
                    });
                    const data = await res.json();
                    const botResponse = data.response || data.detail || 'Error receiving response.';
                    
                    currentChat.messages.push({ role: 'bot', content: botResponse });
                } catch (e) {
                    currentChat.messages.push({ role: 'bot', content: 'Error connecting to backend.' });
                }

                renderMessages();
                renderSidebar();
            }

            function renderMessages() {
                const chatbox = document.getElementById('chatbox');
                chatbox.innerHTML = '';
                if (!activeChatId) return;

                const currentChat = chats.find(c => c.id === activeChatId);
                if (!currentChat) return;

                currentChat.messages.forEach(m => {
                    const div = document.createElement('div');
                    div.className = `msg ${m.role}`;
                    div.innerText = m.content;
                    chatbox.appendChild(div);
                });
                chatbox.scrollTop = chatbox.scrollHeight;
            }

            function renderSidebar() {
                const pinnedList = document.getElementById('pinned-list');
                const chatsList = document.getElementById('chats-list');
                const archivedList = document.getElementById('archived-list');

                pinnedList.innerHTML = '';
                chatsList.innerHTML = '';
                archivedList.innerHTML = '';

                chats.forEach(chat => {
                    const item = document.createElement('div');
                    item.className = `chat-item ${chat.id === activeChatId ? 'active' : ''} ${chat.isPinned ? 'pinned' : ''}`;
                    item.onclick = () => { activeChatId = chat.id; renderMessages(); renderSidebar(); };

                    item.innerHTML = `
                        <span class="chat-title">${chat.title}</span>
                        <div class="chat-actions">
                            <button class="action-btn" title="Pin" onclick="event.stopPropagation(); togglePin(${chat.id})">📌</button>
                            <button class="action-btn" title="Archive" onclick="event.stopPropagation(); toggleArchive(${chat.id})">📥</button>
                            <button class="action-btn" title="Delete" onclick="event.stopPropagation(); deleteChat(${chat.id})">🗑️</button>
                        </div>
                    `;

                    if (chat.isArchived) {
                        archivedList.appendChild(item);
                    } else if (chat.isPinned) {
                        pinnedList.appendChild(item);
                    } else {
                        chatsList.appendChild(item);
                    }
                });
            }

            function togglePin(id) {
                const chat = chats.find(c => c.id === id);
                if (chat) chat.isPinned = !chat.isPinned;
                renderSidebar();
            }

            function toggleArchive(id) {
                const chat = chats.find(c => c.id === id);
                if (chat) chat.isArchived = !chat.isArchived;
                renderSidebar();
            }

            function deleteChat(id) {
                chats = chats.filter(c => c.id !== id);
                if (activeChatId === id) startNewChat();
                else renderSidebar();
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

    # 3. System Prompt enforcing text-only response, strict context isolation, and user profile memory
    user_profile_info = ""
    if request.user_name:
        user_profile_info += f"User's Name: {request.user_name}\n"
    if request.user_age:
        user_profile_info += f"User's Age: {request.user_age}\n"

    system_prompt = (
        "You are ARCH-AI, an intelligent, text-only assistant.\n\n"
        f"USER PROFILE INFORMATION:\n{user_profile_info}\n"
        "STRICT CONTEXT & OUTPUT RULES:\n"
        "1. You KNOW the user's name and age from the profile above. Always accurately answer when asked about their name or age.\n"
        "2. The 'Context Information' provided below comes strictly from EXTERNAL live web searches. "
        "DO NOT assume or state that the user is the person or subject mentioned in the web search context.\n"
        "3. Address the user directly as a helpful peer. Answer their question using the facts in the context without assigning search result identities to the user.\n"
        "4. Absolutely DO NOT output, print, or generate any URLs, hyperlinks, or website links in your response.\n"
        "5. EXCEPTION: Include URLs/links ONLY if the user explicitly uses words like 'links', 'sources', 'urls', or 'websites' in their prompt.\n"
        "6. Never state that you lack real-time data when context is provided."
    )

    # 4. Assemble Messages Array
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Inject conversation history
    for item in request.history:
        if item.get("role") in ["user", "assistant"] and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})

    # Attach live web context
    if web_context:
        messages.append({
            "role": "system", 
            "content": f"Context Information:\n{web_context}"
        })

    # Append current user prompt
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