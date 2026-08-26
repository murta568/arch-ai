import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from tavily import TavilyClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    prompt: str
    history: Optional[List[Message]] = []

class TitleRequest(BaseModel):
    prompt: str

def extract_location(user_prompt: str) -> str:
    """Extracts strictly the city/location name from the prompt."""
    if not groq_client:
        return "Abu Dhabi"
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "Extract ONLY the location or city name mentioned in the prompt. Return plain text only. If no location is mentioned, default to 'Abu Dhabi'."
                },
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=15,
        )
        loc = completion.choices[0].message.content.strip()
        return loc if loc else "Abu Dhabi"
    except Exception:
        return "Abu Dhabi"

def get_live_weather(raw_prompt: str) -> str:
    """Fetches real-time weather globally via WeatherAPI."""
    clean_location = extract_location(raw_prompt)
    
    if WEATHERAPI_KEY:
        try:
            url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY.strip()}&q={clean_location}"
            res = requests.get(url, timeout=5).json()
            
            if "current" in res:
                temp_c = res["current"]["temp_c"]
                feels_c = res["current"].get("feelslike_c", temp_c)
                condition = res["current"]["condition"]["text"]
                city = res["location"]["name"]
                country = res["location"]["country"]
                return f"[LIVE WEATHER API DATA] Real-time weather for {city}, {country}: Actual Temperature: {temp_c}°C, Feels Like: {feels_c}°C, Condition: {condition}."
        except Exception as e:
            print(f"WeatherAPI Connection Error: {e}")

    # Public fallback
    try:
        url = f"https://wttr.in/{clean_location}?format=3"
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and "Unknown" not in res.text:
            return f"[LIVE WEATHER FALLBACK] Real-time weather context: {res.text.strip()}"
    except Exception:
        pass

    return ""

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/title")
def generate_title(request: TitleRequest):
    if not groq_client:
        return {"title": "New Chat"}
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Generate a short 3 to 5 word title for this prompt. Return plain text only with no quotes."},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.5,
            max_tokens=20,
        )
        return {"title": completion.choices[0].message.content.strip()}
    except Exception:
        return {"title": "New Chat"}

@app.post("/chat")
def chat(request: QueryRequest):
    if not groq_client:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY missing. Configure environment variables in Vercel settings."
        )
    
    try:
        context_data = []

        weather_keywords = [
            "weather", "temperature", "forecast", "climate", "rain", 
            "sunny", "hot", "cold", "degree", "temp", "rn", "now", "outside", "today"
        ]
        
        # Always trigger weather lookup if keywords are detected
        if any(keyword in request.prompt.lower() for keyword in weather_keywords):
            weather_info = get_live_weather(request.prompt)
            if weather_info:
                context_data.append(weather_info)

        if tavily_client:
            try:
                search_results = tavily_client.search(query=request.prompt, search_depth="basic")
                results = search_results.get("results", [])
                tavily_text = "\n".join([r.get("content", "") for r in results[:2]])
                if tavily_text:
                    context_data.append(f"Web Context:\n{tavily_text}")
            except Exception:
                pass

        system_instruction = (
            "You are ARCH AI, an intelligent AI assistant. "
            "STRICT INSTRUCTION FOR WEATHER: If real-time weather context is provided, you MUST use the exact numbers provided in the context (Actual Temp, Feels Like, Condition). "
            "NEVER guess, estimate, or modify temperature figures. State the exact figures given."
        )
        
        messages = [{"role": "system", "content": system_instruction}]
        
        if request.history:
            for msg in request.history:
                messages.append({"role": msg.role, "content": msg.content})

        current_prompt = request.prompt
        if context_data:
            current_prompt = "Real-time Context Data:\n" + "\n".join(context_data) + f"\n\nUser Question: {request.prompt}"

        messages.append({"role": "user", "content": current_prompt})

        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.2, # Lower temperature forces strictly accurate responses
            max_tokens=1024,
        )

        return {
            "status": "success",
            "response": completion.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))