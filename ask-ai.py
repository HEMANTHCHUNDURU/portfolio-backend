import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Step 1: Load resume PDF ───────────────────────────────
loader = PyPDFLoader("data/BA_Hemanth Chunduru Resume.pdf")
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# ── Step 2: Chunk it ──────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# ── Step 3: Embed with OpenAI and store in Chroma ─────────
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_storage"
)

print("✅ Resume embedded and stored in chroma_storage/")
print(f"✅ Total chunks stored: {vectordb._collection.count()}")