from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hemanthchunduru.com",
        "https://www.hemanthchunduru.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model — OpenAI (no torch, no download!)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load vector database
vectordb = Chroma(
    persist_directory="./chroma_storage",
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    docs = retriever.invoke(query.question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
You are Hemanth Chunduru's personal AI assistant on his portfolio website.
You help recruiters, hiring managers, and collaborators learn about Hemanth.

Rules:
- Answer from the context provided below.
- Always refer to Hemanth in third person.
- Keep answers concise and professional.
- If someone greets you with ONLY "hi", "hello", "hey" with nothing else,
  respond warmly and briefly introduce yourself.
- For ALL other questions, answer directly without any greeting preamble.
- If someone pastes a Job Description, evaluate Hemanth against it and 
  provide a detailed score out of 10 with breakdown.
- Only say information is not available if completely unrelated to 
  Hemanth's professional background.

Context about Hemanth:
{context}

Question:
{query.question}

Important: If the question contains a job description or asks about fit,
always provide:
1. Requirements MEETS
2. Requirements PARTIALLY meets  
3. Requirements MISSING
4. Score out of 10
5. Overall recommendation
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Hemanth's AI assistant. Answer questions directly and concisely. Only greet back if the user ONLY says hi or hello. For all other questions go straight to the answer. For JD matching always give a score out of 10."},
                {"role": "user", "content": prompt}
            ],
        temperature=0,
        max_tokens=500
    )

    return {"answer": response.choices[0].message.content}