import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re

# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(title="YouTube Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: video_id → chain & metadata
video_chains   = {}
video_metadata = {}


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────
class LoadRequest(BaseModel):
    video_url: str

class ChatRequest(BaseModel):
    video_id: str
    question: str


# ──────────────────────────────────────────────
# Helper: extract YouTube video ID
# ──────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|watch\?v=|&v=)([^#&?\/\s]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract a valid YouTube video ID.")


# ──────────────────────────────────────────────
# Helper: format retrieved docs
# ──────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# ──────────────────────────────────────────────
# Core: build the RAG chain from your notebook
# ──────────────────────────────────────────────
def build_chain(video_id: str):

    # 1. Fetch transcript (from your notebook)
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=["en"]).to_raw_data()
    transcript = " ".join([entry["text"] for entry in transcript_list])

    # 2. Chunk the transcript (from your notebook)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = splitter.create_documents([transcript])

    # 3. Embed + store in ChromaDB (from your notebook)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"yt_{video_id}",
    )

    # 4. Retriever (from your notebook)
    retreiver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 5. Prompt template (from your notebook — exact same text)
    youtube_template = """
You are a Video Research Assistant. Your goal is to answer questions based strictly on the provided YouTube transcript segments.

---
TRANSCRIPT CONTEXT:
{docs}
---

USER QUESTION: {query}

STRICT GUIDELINES:

1. Use ONLY the transcript context above. Do not use outside facts.
2. If the context doesn't mention the answer, say: "I'm sorry, that wasn't mentioned in this video."
3. If the transcript is unstructured (missing punctuation), interpret the flow of speech to provide a coherent answer.
4. Mention the specific part of the video if the context allows.
5.Answer in just 2 sentences maximum.

DETAILED RESPONSE:
"""

    prompt_1 = PromptTemplate(
        template=youtube_template,
        input_variables=["docs", "query"]
    )

    # 6. LLM (from your notebook)
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        task="text-generation",
    )
    model = ChatHuggingFace(llm=llm, temperature=0.7)

    # 7. Final chain (from your notebook)
    parallel_chain = RunnableParallel({
        "docs":  retreiver | RunnableLambda(format_docs),
        "query": RunnablePassthrough(),
    })
    parser = StrOutputParser()
    final_chain = parallel_chain | prompt_1 | model | parser

    return final_chain, transcript


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "loaded_videos": list(video_chains.keys())}


@app.post("/load")
def load_video(req: LoadRequest):
    try:
        video_id = extract_video_id(req.video_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if video_id not in video_chains:
        try:
            chain, transcript = build_chain(video_id)
            video_chains[video_id]   = chain
            video_metadata[video_id] = {
                "transcript_preview": transcript[:300] + "...",
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

    return {
        "video_id":           video_id,
        "message":            "Video loaded successfully! You can now ask questions.",
        "transcript_preview": video_metadata[video_id]["transcript_preview"],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if req.video_id not in video_chains:
        raise HTTPException(status_code=404, detail="Video not loaded. Call /load first.")

    try:
        answer = video_chains[req.video_id].invoke(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return {"answer": answer, "video_id": req.video_id}


@app.delete("/unload/{video_id}")
def unload_video(video_id: str):
    video_chains.pop(video_id, None)
    video_metadata.pop(video_id, None)
    return {"message": f"Video {video_id} unloaded."}


# ──────────────────────────────────────────────
# Serve frontend (put index.html in ./frontend/)
# ──────────────────────────────────────────────
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")



from fastapi.responses import HTMLResponse

@app.get("/")
def root():
    return {"message": "YouTube Chatbot API is running!", "docs": "http://localhost:8000/docs"}
# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)