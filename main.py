from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend domain later (for now allow all during dev)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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

    # 1️⃣ Retrieve relevant docs
    docs = retriever.invoke(query.question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2️⃣ Strict Prompt
    prompt = f"""
You are Hemanth's AI assistant.

Rules:
- Answer ONLY from the resume context below.
- If information is not present, say:
  "That information is not available in Hemanth's resume."
- Keep answers concise and professional.

Resume Context:
{context}

Question:
{query.question}
"""

    # 3️⃣ Call OpenAI (low cost model)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions about Hemanth's resume."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=250
    )

    answer = response.choices[0].message.content

    return {"answer": answer}
