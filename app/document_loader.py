import os
from app.config import DATA_FOLDER


def load_documents(folder=DATA_FOLDER):
    documents = {}

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()

    return documents