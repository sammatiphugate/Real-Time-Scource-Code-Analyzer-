import shutil
import subprocess
import threading
import uuid
from urllib.parse import urlparse

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.paths import PROJECT_ROOT, REPO_DIR

_clone_lock = threading.Lock()


def _parse_github_repo(repo_url: str):
    parsed = urlparse(repo_url.strip().rstrip("/"))
    host = (parsed.hostname or "").lower()
    if host not in ("github.com", "www.github.com"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1].removesuffix(".git")
    return owner, repo


def validate_github_repo(repo_url: str) -> None:
    """Check repo exists via git (avoids GitHub REST API 403 rate limits)."""
    parsed = _parse_github_repo(repo_url)
    owner, repo = (parsed if parsed else (None, None))

    proc = subprocess.run(
        ["git", "ls-remote", "--heads", repo_url],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return

    err = (proc.stderr or proc.stdout or "").strip()
    if "not found" in err.lower() or "Repository not found" in err:
        hint = ""
        if repo == "brain-breast":
            hint = ' Did you mean "https://github.com/d-hackmt/brain-breast-cancer"?'
        raise ValueError(
            f'GitHub repository "{owner}/{repo}" was not found. '
            f"Use the full public URL with exact spelling.{hint}"
        )
    raise ValueError(
        f"Cannot access repository. Check the URL and your network. {err[:300]}".strip()
    )


def _format_clone_error(stderr: str, stdout: str) -> str:
    text = (stderr or stdout or "").strip()
    if "Repository not found" in text or "not found" in text.lower():
        return (
            "GitHub repository not found. Verify the full URL "
            "(example: https://github.com/d-hackmt/brain-breast-cancer)."
        )
    if "could not lock config file" in text or "unable to write file" in text:
        return (
            "Git clone was interrupted (folder locked or removed mid-clone). "
            "Wait a few seconds and click Index again. If it keeps failing, move the "
            "project out of Desktop/iCloud or grant Full Disk Access to your terminal."
        )
    if len(text) > 500:
        text = text[:500] + "…"
    return f"git clone failed: {text}" if text else "git clone failed"


def repo_ingestion(repo_url: str) -> None:
    """Clone into a temp folder, then move into repo/ (avoids partial .git races)."""
    repo_url = repo_url.strip()
    validate_github_repo(repo_url)

    with _clone_lock:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)

        tmp_dir = PROJECT_ROOT / f".repo_clone_{uuid.uuid4().hex}"
        try:
            proc = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    repo_url,
                    str(tmp_dir),
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(_format_clone_error(proc.stderr, proc.stdout))

            if not (tmp_dir / ".git").is_dir():
                raise RuntimeError("git clone finished but .git folder is missing")

            tmp_dir.rename(REPO_DIR)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if REPO_DIR.exists():
                shutil.rmtree(REPO_DIR, ignore_errors=True)
            raise


def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language="python", parser_threshold=500),
    )
    documents = loader.load()
    return documents


def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python",
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks


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
    documents = load_repo(str(REPO_DIR))

    print("Splitting text...")
    chunks = text_splitter(documents)

    print("Loading embeddings...")
    embeddings = load_embedding()

    print("✅ Done!")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
