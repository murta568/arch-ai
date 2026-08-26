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

def get_accurate_weather(raw_prompt: str) -> str:
    """Uses Geocoding + Open-Meteo for high-accuracy real-time weather (No API key needed)."""
    try:
        # 1. Geocode location string to lat/long
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name=Abu%20Dhabi&count=1&language=en&format=json"
        
        # Extract specific city if present in prompt
        if "dubai" in raw_prompt.lower():
            geo_url = "https://geocoding-api.open-meteo.com/v1/search?name=Dubai&count=1&language=en&format=json"
        elif "sharjah" in raw_prompt.lower():
            geo_url = "https://geocoding-api.open-meteo.com/v1/search?name=Sharjah&count=1&language=en&format=json"
            
        geo_res = requests.get(geo_url, timeout=5).json()
        
        if "results" in geo_res and len(geo_res["results"]) > 0:
            lat = geo_res["results"][0]["latitude"]
            lon = geo_res["results"][0]["longitude"]
            city = geo_res["results"][0]["name"]
            country = geo_res["results"][0].get("country", "")

            # 2. Query Open-Meteo Current Weather
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,weather_code&timezone=auto"
            w_res = requests.get(weather_url, timeout=5).json()

            if "current" in w_res:
                temp_c = w_res["current"]["temperature_2m"]
                feels_c = w_res["current"]["apparent_temperature"]
                return f"[LIVE ACCURATE DATA] Current real-time weather for {city}, {country}: Actual Temp: {temp_c}°C, Feels Like: {feels_c}°C."
    except Exception as e:
        print(f"Weather error: {e}")

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
        
        if any(keyword in request.prompt.lower() for keyword in weather_keywords):
            weather_info = get_accurate_weather(request.prompt)
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
            "STRICT WEATHER INSTRUCTION: Use ONLY the exact numbers provided in [LIVE ACCURATE DATA]. "
            "Do NOT hallucinate or guess temperatures. State the exact actual and feels-like readings provided."
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
            temperature=0.1,
            max_tokens=1024,
        )

        return {
            "status": "success",
            "response": completion.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))