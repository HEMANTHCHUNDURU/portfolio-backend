import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Load ALL PDFs from data/ folder ──────────────────────
loader = PyPDFDirectoryLoader("data/")
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages from {len(set(d.metadata['source'] for d in documents))} files")

# ── Chunk ─────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# ── Embed and store ───────────────────────────────────────
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Delete old chroma_storage first
import shutil
if os.path.exists("./chroma_storage"):
    shutil.rmtree("./chroma_storage")
    print("✅ Cleared old chroma_storage")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_storage"
)

print("✅ Both resumes embedded and stored in chroma_storage/")
print(f"✅ Total chunks stored: {vectordb._collection.count()}")