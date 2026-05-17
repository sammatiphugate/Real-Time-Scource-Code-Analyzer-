# Real-Time-Scource-Code-Analyzer-
In this project, i  have  build a Source Code Analyzer, a powerful tool that acts as a senior dIeveloper paired with you to navigate complex codebases. 

# Source-Code-Analysis-Project


# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n llmapp python=3.10 -y
```

```bash
conda activate llmapp
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Groq API key:

```ini
GROQ_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
Then open the app in your browser:

```
http://localhost:8080
```


### Techstack Used:

- Python
- LangChain
- Flask
- Groq (LLM inference)
- HuggingFace Sentence Transformers (embeddings)
- ChromaDB

### Documentation

See `docs/PROJECT_REPORT.md` for the full project write-up (architecture, diagrams, testing, etc.).