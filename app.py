from fastapi import FastAPI
from pydantic import BaseModel
from rag import final_chain
import dotenv

dotenv.load_dotenv(".env")

app = FastAPI(title="RAG Chatbot API")

# Request model
class ChatRequest(BaseModel):
    question: str

# Response model
class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_input = request.question

    if user_input.lower() == "exit":
        return {"answer": "Session ended"}

    # Invoke your RAG chain
    result = final_chain.invoke(user_input)

    return {"answer": result}


@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running. Use POST /chat to ask questions."}
