import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from tavily import TavilyClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

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

class QueryRequest(BaseModel):
    prompt: str

class TitleRequest(BaseModel):
    prompt: str

def get_live_weather(location_query: str) -> str:
    """Fetches real-time weather data for any location."""
    if OPENWEATHER_API_KEY:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location_query}&appid={OPENWEATHER_API_KEY}&units=metric"
            res = requests.get(url, timeout=5).json()
            if res.get("cod") == 200:
                temp = res["main"]["temp"]
                desc = res["weather"][0]["description"]
                city = res["name"]
                country = res["sys"]["country"]
                return f"Live weather for {city}, {country}: {temp}°C, {desc}."
        except Exception:
            pass

    # Fallback to no-key live weather provider
    try:
        url = f"https://wttr.in/{location_query}?format=3"
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and "Unknown" not in res.text:
            return f"Live weather context: {res.text.strip()}"
    except Exception:
        pass

    return ""

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/title")
def generate_title(request: TitleRequest):
    """Generates a short 3-5 word chat title like ChatGPT/Claude."""
    if not groq_client:
        return {"title": "New Chat"}
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Generate a concise 3 to 5 word title for this prompt. Return ONLY the title text, no quotes or labels."},
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
            detail="GROQ_API_KEY missing. Ensure it is configured in your environment."
        )
    
    try:
        context_data = []

        # Weather Detection & Fetching
        weather_keywords = ["weather", "temperature", "forecast", "climate", "rain", "sunny", "hot", "cold"]
        if any(keyword in request.prompt.lower() for keyword in weather_keywords):
            weather_info = get_live_weather(request.prompt)
            if weather_info:
                context_data.append(weather_info)

        # Web Search Context
        if tavily_client:
            try:
                search_results = tavily_client.search(query=request.prompt, search_depth="basic")
                results = search_results.get("results", [])
                tavily_text = "\n".join([r.get("content", "") for r in results[:2]])
                if tavily_text:
                    context_data.append(f"Web Context:\n{tavily_text}")
            except Exception:
                pass

        system_instruction = "You are ARCH AI, a helpful, intelligent assistant with access to real-time weather and web data."
        full_prompt = request.prompt
        if context_data:
            full_prompt = f"Real-time Context:\n" + "\n".join(context_data) + f"\n\nUser Question: {request.prompt}"

        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        return {
            "status": "success",
            "response": completion.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))