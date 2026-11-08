from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.retrieval.store import close_client


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    close_client()  # close the shared Weaviate connection on shutdown


app = FastAPI(title="Tesseract v2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}