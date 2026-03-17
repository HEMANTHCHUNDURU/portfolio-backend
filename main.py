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
You are Hemanth's AI assistant on his portfolio website.

Rules:
- Answer ONLY from the resume context below.
- If information is not present, say:
  "That information is not available in Hemanth's resume."
- Keep answers concise and professional.
- Write in third person about Hemanth.

Resume Context:
{context}

Question:
{query.question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions about Hemanth's resume and background."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=250
    )

    return {"answer": response.choices[0].message.content}