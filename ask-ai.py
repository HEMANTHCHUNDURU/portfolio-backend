import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Step 1: Load your resume PDF ──────────────────────────
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

# ── Step 3: Embed and store in Chroma ─────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_storage"
)

print("✅ Resume embedded and stored in chroma_storage/")
print(f"✅ Total chunks stored: {vectordb._collection.count()}")
