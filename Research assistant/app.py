from flask import Flask, render_template, request, jsonify
import os, numpy as np

from utils.pdf_loader import extract_text_from_pdf
from utils.embedder import embed_text
from utils.retriever import load_index, save_index, load_texts, save_texts
from utils.lm_client import ask_lm
from config import UPLOAD_FOLDER, CHUNK_SIZE

app = Flask(__name__)

index = load_index()
texts = load_texts()

@app.route("/")
def home():
    return render_template("index.html")

# ------------------------ UPLOAD -------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = extract_text_from_pdf(path)

    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    for chunk in chunks:
        vec = embed_text(chunk)
        index.add(np.array([vec]))
        texts.append(chunk)

    save_index(index)
    save_texts(texts)

    return jsonify({"message": "PDF uploaded & indexed!"})

# ------------------------- CHAT --------------------------
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json["message"]

    q_vec = embed_text(msg)

    D, I = index.search(np.array([q_vec]), 3)
    retrieved = "\n\n".join([texts[i] for i in I[0]])

    prompt = f"""
You are a research assistant.
Use the context below to answer:

Context:
{retrieved}

Question: {msg}
Answer:
"""

    response = ask_lm(prompt)
    return jsonify({"reply": response})

app.run(debug=True)
