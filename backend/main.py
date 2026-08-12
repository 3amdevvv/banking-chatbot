from pathlib import Path
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import BankingRAG
from gemini import generate_answer


# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="Banking Chatbot API",
    description="RAG chatbot using FastAPI and Gemini",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=["https://upi-chatbot.netlify.app"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "data_small.json"
)


# --------------------------------------------------
# LOAD RAG SYSTEM
# --------------------------------------------------

print("Loading banking dataset...")

rag = BankingRAG(
    DATA_PATH
)

print("Banking chatbot is ready!")


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):

    message: str

    top_k: int = 5
    conversation: list = []


# --------------------------------------------------
# RESPONSE MODEL
# --------------------------------------------------

class ChatResponse(BaseModel):

    answer: str

    sources: list


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Banking Chatbot API is running",
        "status": "success"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "documents": len(rag.documents)
    }


# --------------------------------------------------
# SEARCH ENDPOINT
# --------------------------------------------------

@app.get("/search")
def search_dataset(
    q: str,
    top_k: int = 5
):

    if not q.strip():

        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    results = rag.search(
        q,
        top_k
    )

    sources = []

    for result in results:

        sources.append({
            "question": result["user_query"],
            "category": result["domain_category"],
            "subdomain": result["subdomain"],
            "score": result["score"]
        })

    return {
        "query": q,
        "results": sources
    }


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    user_message = request.message.strip()

    if not user_message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # ----------------------------------------------
    # STEP 1: Retrieve relevant records
    # ----------------------------------------------

    documents = rag.search(
        user_message,
        top_k=request.top_k
    )

    # ----------------------------------------------
    # STEP 2: Generate Gemini response
    # ----------------------------------------------

    try:

        answer = generate_answer(
            user_question=user_message,
            retrieved_documents=documents,
            conversation=request.conversation
        )

    except Exception as e:

        print("Gemini error:", e)

        raise HTTPException(
            status_code=500,
            detail="Failed to generate chatbot response."
        )

    # ----------------------------------------------
    # STEP 3: Return response
    # ----------------------------------------------

    sources = []

    for document in documents:

        sources.append({
            "question": document["user_query"],
            "category": document["domain_category"],
            "subdomain": document["subdomain"],
            "score": round(
                document["score"],
                4
            )
        })

    return ChatResponse(
        answer=answer,
        sources=sources
    )
