import os
from typing import TypedDict, Literal, Annotated
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 1. Setup Tavily Search Tool (Updated to langchain-tavily)
tavily_raw_tool = TavilySearch(max_results=2)

@tool
def web_search(query: str) -> str:
    """Search the web for news, facts, or current events."""
    raw_results = tavily_raw_tool.invoke({"query": query})
    results_str = str(raw_results)
    return results_str[:1500] 

tools = [web_search]

# 2. Setup LLM
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_INSTRUCTION = SystemMessage(
    content="""You are a concise AI search assistant.
1. When using web_search, pass a short search string as the 'query' argument.
2. Keep your search requests focused and brief.
3. Synthesize the findings into a clear, direct answer."""
)

# 3. Define Graph State
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. Agent Node
def agent_node(state: GraphState):
    messages = state["messages"]
    
    # Keep only the last 6 messages to stay within token limits
    trimmed_messages = messages[-6:] if len(messages) > 6 else messages
    
    if not isinstance(trimmed_messages[0], SystemMessage):
        trimmed_messages = [SYSTEM_INSTRUCTION] + trimmed_messages
        
    response = llm_with_tools.invoke(trimmed_messages)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n[AGENT] 🌐 Web Search Query: {tool_call['args']}\n")
            
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 5. Routing Logic
def should_continue(state: GraphState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 6. Workflow Setup
workflow = StateGraph(GraphState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

langgraph_app = workflow.compile()

# 7. FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/ask")
def ask(question: str):
    try:
        initial_state = {
            "messages": [HumanMessage(content=question)]
        }
        
        result = langgraph_app.invoke(initial_state)
        final_response = result["messages"][-1].content
        
        used_search = any(getattr(msg, "type", None) == "tool" for msg in result["messages"])
        
        return {
            "question": question, 
            "answer": final_response,
            "used_web_search": used_search
        }
    except Exception as e:
        print(f"Error encountered: {e}")
        return {
            "question": question,
            "answer": f"Error handling request: {str(e)}",
            "used_web_search": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)