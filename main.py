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
from zeep import Client as SoapClient

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
    username: Optional[str] = "User"
    history: Optional[List[Message]] = []

class TitleRequest(BaseModel):
    prompt: str
    response_text: Optional[str] = ""

# --- 1. WEATHER REST API FUNCTION ---
def get_accurate_weather(raw_prompt: str) -> str:
    """REST API: Open-Meteo Geocoding and Weather"""
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?name=Abu%20Dhabi&count=1&language=en&format=json"
        
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

            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,weather_code&timezone=auto"
            w_res = requests.get(weather_url, timeout=5).json()

            if "current" in w_res:
                temp_c = w_res["current"]["temperature_2m"]
                feels_c = w_res["current"]["apparent_temperature"]
                return f"[LIVE WEATHER REST DATA] Weather for {city}, {country}: Actual Temp: {temp_c}°C, Feels Like: {feels_c}°C."
    except Exception as e:
        print(f"Weather error: {e}")

    return ""

# --- 2. REST API EXAMPLE (JSONPlaceholder) ---
def fetch_rest_sample_data() -> str:
    """REST API: Fetches sample data from JSONPlaceholder"""
    try:
        res = requests.get("https://jsonplaceholder.typicode.com/todos/1", timeout=3)
        if res.status_code == 200:
            data = res.json()
            return f"[REST API DATA] Sample task fetched: Title: '{data.get('title')}', Completed: {data.get('completed')}."
    except Exception as e:
        print(f"REST API error: {e}")
    return ""

# --- 3. SOAP API EXAMPLE (DataFlex NumberConversion) ---
def call_soap_number_to_words(number: int = 250) -> str:
    """SOAP API: Converts a number into words using WSDL XML web service"""
    try:
        wsdl_url = "https://www.dataaccess.com/webservicesserver/numberconversion.wso?WSDL"
        soap_client = SoapClient(wsdl=wsdl_url)
        result = soap_client.service.NumberToWords(ubiNum=number)
        return f"[SOAP API DATA] Number {number} in words via SOAP Web Service: '{result.strip()}'."
    except Exception as e:
        print(f"SOAP API error: {e}")
        return ""

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/title")
def generate_title(request: TitleRequest):
    if not groq_client:
        return {"title": "New Chat"}
    try:
        user_content = f"User asked: {request.prompt}\nAI answered: {request.response_text}"
        
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system", 
                    "content": "Create a brief 2 to 4 word summary title for this conversation. Output ONLY the title text. Do not use quotes, punctuation, or extra words."
                },
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=15,
        )
        return {"title": completion.choices[0].message.content.strip()}
    except Exception as e:
        print(f"Title Error: {e}")
        return {"title": request.prompt[:20]}

@app.post("/chat")
def chat(request: QueryRequest):
    if not groq_client:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY missing. Configure environment variables in Vercel settings."
        )
    
    try:
        context_data = []
        prompt_lower = request.prompt.lower()

        # Trigger Weather REST API
        weather_keywords = ["weather", "temperature", "forecast", "climate", "rain", "sunny", "hot", "cold", "temp", "today"]
        is_weather_query = any(keyword in prompt_lower for keyword in weather_keywords)
        if is_weather_query:
            weather_info = get_accurate_weather(request.prompt)
            if weather_info:
                context_data.append(weather_info)

        # Trigger Sample REST API
        if "rest" in prompt_lower or "sample data" in prompt_lower or "todo" in prompt_lower:
            rest_data = fetch_rest_sample_data()
            if rest_data:
                context_data.append(rest_data)

        # Trigger SOAP API
        if "soap" in prompt_lower or "wsdl" in prompt_lower or "words" in prompt_lower:
            soap_data = call_soap_number_to_words(250)
            if soap_data:
                context_data.append(soap_data)

        # AUTOMATIC TAVILY WEB SEARCH
        # Searches the web for any query unless it's a specific internal command or brief greeting
        skip_search = any(k in prompt_lower for k in ["hi", "hello", "hey"]) and len(prompt_lower.split()) < 3
        if tavily_client and not skip_search:
            try:
                search_results = tavily_client.search(query=request.prompt, search_depth="basic")
                results = search_results.get("results", [])
                tavily_text = "\n".join([r.get("content", "") for r in results[:2]])
                if tavily_text:
                    context_data.append(f"Web Context:\n{tavily_text}")
            except Exception as e:
                print(f"Tavily Search Error: {e}")

        user_name = request.username if request.username else "User"

        system_instruction = (
            f"You are ARCH AI, an intelligent AI assistant. "
            f"The user's name is '{user_name}'. ALWAYS remember their name and address them by name when asked or appropriate. "
            "STRICT INSTRUCTION: Use any live REST, SOAP, or Web Context data provided to give precise, up-to-date responses."
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