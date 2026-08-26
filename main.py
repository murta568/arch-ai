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

class QueryRequest(BaseModel):
    prompt: str

class TitleRequest(BaseModel):
    prompt: str

def extract_location(user_prompt: str) -> str:
    """Extracts only the city/location name from the user's prompt."""
    if not groq_client:
        return user_prompt
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "Extract ONLY the location or city name mentioned in the prompt. Return plain text only. If no location is mentioned, return 'Abu Dhabi'."
                },
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=15,
        )
        extracted = completion.choices[0].message.content.strip()
        return extracted if extracted else "Abu Dhabi"
    except Exception:
        return "Abu Dhabi"

def get_live_weather(raw_prompt: str) -> str:
    """Forces an explicit live WeatherAPI call using extracted location."""
    clean_location = extract_location(raw_prompt)
    
    # Priority 1: WeatherAPI Call
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
                return f"[LIVE DATA] Real-time weather for {city}, {country}: Actual Temp: {temp_c}°C, Feels Like: {feels_c}°C, Condition: {condition}."
            elif "error" in res:
                print(f"WeatherAPI Error: {res['error'].get('message')}")
        except Exception as e:
            print(f"WeatherAPI Connection Error: {e}")

    # Priority 2: wttr.in fallback if key is missing/failed
    try:
        url = f"https://wttr.in/{clean_location}?format=3"
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and "Unknown" not in res.text:
            return f"[FALLBACK DATA] Weather context: {res.text.strip()}"
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

        # Keywords to force live weather search
        weather_keywords = [
            "weather", "temperature", "forecast", "climate", "rain", 
            "sunny", "hot", "cold", "degree", "temp", "rn", "now", "outside"
        ]
        
        # Check if user prompt is asking about weather/temperature
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
            "You MUST use the real-time context provided to answer weather questions. "
            "Do NOT output function calls, JSON payloads, or code snippets. Respond using simple text."
        )
        
        full_prompt = request.prompt
        if context_data:
            full_prompt = "Real-time Context Data:\n" + "\n".join(context_data) + f"\n\nUser Question: {request.prompt}"

        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,
            max_tokens=1024,
        )

        return {
            "status": "success",
            "response": completion.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))