from app.retrieval.search import search

for hit in search("Interstellar_Script", "What is the tesseract?"):
    print(round(hit["distance"], 3), hit["text"][:100])