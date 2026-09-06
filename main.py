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
        if any(keyword in prompt_lower for keyword in weather_keywords):
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
        skip_search = any(k in prompt_lower for k in ["hi", "hello", "hey"]) and len(prompt_lower.split()) < 3
        if tavily_client and not skip_search:
            try:
                search_results = tavily_client.search(query=request.prompt, search_depth="basic")
                results = search_results.get("results", [])
                tavily_text = "\n".join([r.get("content", "") for r in results[:3]])
                if tavily_text:
                    context_data.append(f"Web Context:\n{tavily_text}")
            except Exception as e:
                print(f"Tavily Search Error: {e}")

        user_name = request.username if request.username else "User"

        system_instruction = (
            f"You are ARCH AI, an intelligent AI assistant. "
            f"The user's name is '{user_name}'. ALWAYS remember their name and address them by name when asked or appropriate. "
            "INSTRUCTION: Use the provided 'Real-time Context Data' to answer questions about live prices, scores, news, or weather. "
            "Do NOT state that you lack a live market or sports feed if web context is present in the prompt."
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