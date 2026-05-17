from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import gc
import os
import shutil
import time
from src.helper import load_embedding, repo_ingestion
from src.paths import REPO_DIR, DB_DIR, PROJECT_ROOT
from pathlib import Path

# ✅ Updated imports
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain

# ✅ CHANGE 1: Added these imports needed for /index route
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


# ==============================
# Flask App
# ==============================
app = Flask(__name__)


# ==============================
# Load ENV
# ==============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==============================
# Load Embeddings + Vector DB
# ==============================
embeddings = load_embedding()

vectordb = None
qa = None
memory = None


# ==============================
# LLM (Groq)
# ==============================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)


def _chroma_db_ready() -> bool:
    return (DB_DIR / "chroma.sqlite3").is_file()


def _close_vectordb() -> None:
    """Release Chroma/SQLite handles before deleting db/ (prevents readonly DB errors)."""
    global vectordb
    if vectordb is None:
        return
    try:
        client = getattr(vectordb, "_client", None)
        if client is not None and hasattr(client, "clear_system_cache"):
            client.clear_system_cache()
    except Exception:
        pass
    vectordb = None
    gc.collect()
    time.sleep(0.1)


def _prepare_db_dir() -> None:
    _close_vectordb()
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DB_DIR, 0o755)


def _build_qa_chain(store: Chroma):
    global memory
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3},
        ),
        memory=memory,
    )


def _load_vectordb_from_disk() -> None:
    global vectordb, qa
    if not _chroma_db_ready():
        vectordb = None
        qa = None
        return
    vectordb = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    qa = _build_qa_chain(vectordb)


_load_vectordb_from_disk()


def _safe_ai_text(result):
    """Normalize LangChain/Groq result object into plain text."""
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(result, str):
        return result
    return str(result)


def _read_repo_file(rel_path: str) -> str:
    """Read a file only from inside ./repo safely."""
    repo_root = REPO_DIR.resolve()
    target = (repo_root / (rel_path or "")).resolve()
    if repo_root not in target.parents and target != repo_root:
        raise ValueError("Invalid file path")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError("File not found")
    return target.read_text(encoding="utf-8", errors="replace")


def _run_action_prompt(action: str, code: str, file_path: str = "", target_language: str = "") -> str:
    """Execute a structured action prompt via Groq model."""
    trimmed = (code or "").strip()
    if not trimmed:
        raise ValueError("Code is required")

    prompts = {
        "explain": "Explain this code clearly with sections: Purpose, Key Logic, Edge Cases, and Improvements.",
        "optimize": "Optimize this code for readability and performance. Return:\n1) Issues\n2) Improved code block\n3) Why changes help",
        "find-bugs": "Find likely bugs and risky patterns. Return severity-tagged bullets and fixes.",
        "comment": "Add meaningful comments/docstrings to this code. Return commented code in a code block.",
    }
    if action == "convert":
        lang = (target_language or "javascript").strip()
        instruction = f"Convert this code to {lang}. Keep behavior equivalent. Return only the converted code in one fenced block."
    else:
        instruction = prompts.get(action)

    if not instruction:
        raise ValueError("Unsupported action")

    prompt = (
        f"Action: {action}\n"
        f"File: {file_path or 'N/A'}\n\n"
        f"{instruction}\n\n"
        "Code:\n```text\n"
        f"{trimmed[:28000]}\n"
        "```"
    )
    return _safe_ai_text(llm.invoke(prompt))


# ==============================
# Routes
# ==============================

# ✅ CHANGE 2: render 'chat.html' instead of 'index.html'
#    (HTML file from Blueprint Studio design is saved as chat.html)
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')  # was: index.html


# ✅ CHANGE 3: Replaced old /chatbot route with /index route
#    Old /chatbot: took 'question' field, ran store_index.py as subprocess
#    New  /index:  takes 'repo_url' field, clones + indexes inline,
#                  returns {"status","files"} JSON for sidebar population in HTML
@app.route('/index', methods=["POST"])  # was: /chatbot
def index_repo():
    repo_url = (request.form.get("repo_url") or "").strip()  # was: request.form['question']

    if not repo_url:
        return jsonify({"status": "error", "message": "repo_url is required"}), 400

    try:
        print("Cloning repo:", repo_url)

        repo_ingestion(repo_url)

        _prepare_db_dir()

        # ✅ CHANGE 5: Replaced os.system("python store_index.py") with
        #    inline loading + splitting + embedding so we can:
        #    a) get the file list to send back to the HTML sidebar
        #    b) rebuild the global qa chain immediately after indexing
        loader = GenericLoader.from_filesystem(
            str(REPO_DIR),
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        )
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=500,
            chunk_overlap=20
        )
        texts = splitter.split_documents(documents)

        if not texts:
            return jsonify(
                {"status": "error", "message": "No Python files found in this repository."}
            ), 400

        # Embed into ChromaDB
        global vectordb, qa
        vectordb = Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory=str(DB_DIR),
        )
        qa = _build_qa_chain(vectordb)

        # ✅ CHANGE 7: Build file list to send back to HTML
        #    HTML sidebar uses this to show indexed files in left panel
        files = [
            os.path.relpath(doc.metadata["source"], str(REPO_DIR))
            for doc in documents
        ]
        files = list(dict.fromkeys(files))  # deduplicate

        print(f"Indexed {len(texts)} chunks from {len(files)} files")

        # ✅ CHANGE 8: Return JSON with status + files (HTML expects this format)
        #    was: jsonify({"response": "Repository processed successfully!"})
        return jsonify({"status": "ok", "files": files})

    except Exception as e:
        print(f"Indexing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# 🔥 Chat Endpoint — mostly unchanged
@app.route("/get", methods=["POST"])
def chat():
    user_input = (request.form.get("msg") or "").strip()

    if not user_input:
        return "Message is required.", 400

    print("User:", user_input)

    if user_input.lower() == "clear":
        global vectordb, qa
        _close_vectordb()
        qa = None
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        if DB_DIR.exists():
            shutil.rmtree(DB_DIR)
        return "Cleared repository and database."

    if qa is None:
        return (
            "No indexed repository yet. Paste a GitHub URL and click "
            '"Index Repository", then ask your question again.'
        )

    try:
        result = qa({"question": user_input})
    except Exception as exc:
        err = str(exc)
        if "readonly database" in err.lower() or "1032" in err:
            return (
                "Vector database is locked or corrupted. Click Index Repository again "
                "to rebuild it, or type 'clear' and re-index."
            )
        raise

    print("Bot:", result["answer"])

    return str(result["answer"])


@app.route("/file-content", methods=["POST"])
def file_content():
    file_path = (request.form.get("file_path") or "").strip()
    if not file_path:
        return jsonify({"status": "error", "message": "file_path is required"}), 400
    try:
        content = _read_repo_file(file_path)
        return jsonify({"status": "ok", "file_path": file_path, "content": content})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found in indexed repo"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


def _action_response(action_name: str):
    file_path = (request.form.get("file_path") or "").strip()
    selected_code = request.form.get("selected_code")
    full_code = request.form.get("file_content")
    target_language = request.form.get("target_language", "")
    code = selected_code if (selected_code or "").strip() else full_code

    if not file_path and not (code or "").strip():
        return jsonify({"status": "error", "message": "file_path or code is required"}), 400
    try:
        if not (code or "").strip() and file_path:
            code = _read_repo_file(file_path)
        output = _run_action_prompt(action_name, code, file_path=file_path, target_language=target_language)
        return jsonify(
            {
                "status": "ok",
                "action": action_name,
                "file_path": file_path,
                "used_selection": bool((selected_code or "").strip()),
                "result": output,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    return _action_response("explain")


@app.route("/optimize", methods=["POST"])
def optimize():
    return _action_response("optimize")


@app.route("/find-bugs", methods=["POST"])
def find_bugs():
    return _action_response("find-bugs")


@app.route("/convert", methods=["POST"])
def convert():
    return _action_response("convert")


@app.route("/comment", methods=["POST"])
def comment():
    return _action_response("comment")


# ==============================
# Run App
# ==============================
if __name__ == '__main__':
    os.chdir(PROJECT_ROOT)
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)