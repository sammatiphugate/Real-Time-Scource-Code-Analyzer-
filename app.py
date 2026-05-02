from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from git import Repo
import shutil
from src.helper import load_embedding, repo_ingestion
from pathlib import Path

# ✅ Updated imports
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

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

persist_directory = "db"

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)


# ==============================
# LLM (Groq)
# ==============================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)


# ==============================
# Memory
# ==============================
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)


# ==============================
# QA Chain
# ==============================
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    ),
    memory=memory
)


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
    repo_root = Path("repo").resolve()
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
        # ✅ CHANGE 4: Using shutil.rmtree instead of os.system("rm -rf ...")
        #    shutil works on Windows, Mac, and Linux — os.system rm does not
        if os.path.exists("repo"):
            shutil.rmtree("repo")
        if os.path.exists("db"):
            shutil.rmtree("db")

        print("Cloning repo:", repo_url)

        # Clone repo (same logic as before via repo_ingestion)
        repo_ingestion(repo_url)

        # ✅ CHANGE 5: Replaced os.system("python store_index.py") with
        #    inline loading + splitting + embedding so we can:
        #    a) get the file list to send back to the HTML sidebar
        #    b) rebuild the global qa chain immediately after indexing
        loader = GenericLoader.from_filesystem(
            "repo",
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

        # Embed into ChromaDB
        new_vectordb = Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory="./db"
        )

        # ✅ CHANGE 6: Rebuild qa chain globally so /get uses the new repo's DB
        #    Without this, chat would still answer from the old vector store
        global qa, memory
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=new_vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            ),
            memory=memory
        )

        # ✅ CHANGE 7: Build file list to send back to HTML
        #    HTML sidebar uses this to show indexed files in left panel
        files = [
            os.path.relpath(doc.metadata["source"], "repo")
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
        # ✅ CHANGE 9: Replaced os.system("rm -rf repo db") with shutil.rmtree
        #    Same reason as CHANGE 4 — cross-platform safety
        if os.path.exists("repo"):
            shutil.rmtree("repo")
        if os.path.exists("db"):
            shutil.rmtree("db")
        return "Cleared repository and database."

    # Unchanged — same qa call as before
    result = qa({"question": user_input})

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
    app.run(host="0.0.0.0", port=8080, debug=True)