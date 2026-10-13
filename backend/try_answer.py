import sys

from app.generation.generator import generate
from app.retrieval.search import search

question = " ".join(sys.argv[1:]) or "What is the tesseract in Interstellar?"
hits = search("Interstellar_Script", question)
print(generate(question, [h["text"] for h in hits]))
