from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import os


# ==============================
# Load Environment Variables
# ==============================
load_dotenv()


# ==============================
# Step 1: Clone Repo (if not exists)
# ==============================
#repo_url = "https://github.com/entbappy/End-to-end-Medical-Chatbot-Generative-AI"

if not os.path.exists("repo/"):
    print("Cloning repository...")
    #repo_ingestion(repo_url)


# ==============================
# Step 2: Load Documents
# ==============================
print("Loading repository...")
documents = load_repo("repo/")


# ==============================
# Step 3: Split into Chunks
# ==============================
print("Splitting text...")
text_chunks = text_splitter(documents)


# ==============================
# Step 4: Load Embeddings (HuggingFace)
# ==============================
print("Loading embeddings...")
embeddings = load_embedding()


# ==============================
# Step 5: Store in Chroma DB
# ==============================
print("Storing in vector database...")

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./db"
)

vectordb.persist()


print("✅ Vector DB created successfully!")
print(f"Total Documents: {len(documents)}")
print(f"Total Chunks: {len(text_chunks)}")