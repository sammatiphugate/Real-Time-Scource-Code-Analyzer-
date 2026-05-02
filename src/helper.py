import os
from git import Repo

# ✅ Updated imports (new LangChain structure)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # was: from langchain.text_splitter
from langchain_huggingface import HuggingFaceEmbeddings              # was: from langchain_community.embeddings



# ==============================
# Clone GitHub Repository
# ==============================
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)


# ==============================
# Load Repository Files
# ==============================
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language="python", parser_threshold=500)  # ✅ FIX
    )

    documents = loader.load()
    return documents


# ==============================
# Split Text into Chunks
# ==============================
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python",   # ✅ FIX (no Language.PYTHON)
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks


# ==============================
# Load Embeddings (HuggingFace)
# ==============================
def load_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


if __name__ == "__main__":
    repo_url = "https://github.com/entbappy/End-to-end-Medical-Chatbot-Generative-AI"

    print("Cloning repo...")
    repo_ingestion(repo_url)

    print("Loading repo...")
    documents = load_repo("repo/")

    print("Splitting text...")
    chunks = text_splitter(documents)

    print("Loading embeddings...")
    embeddings = load_embedding()

    print("✅ Done!")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")