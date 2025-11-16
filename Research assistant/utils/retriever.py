import faiss
import json
import numpy as np
import os

INDEX_PATH = "data/faiss_index.bin"
TEXT_PATH = "data/texts.json"

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(384)

def save_index(index):
    faiss.write_index(index, INDEX_PATH)

def load_texts():
    if os.path.exists(TEXT_PATH):
        return json.load(open(TEXT_PATH, "r"))
    return []

def save_texts(texts):
    json.dump(texts, open(TEXT_PATH, "w"))