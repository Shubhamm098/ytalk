import os
import re
import requests
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

video_chains   = {}
video_metadata = {}

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
class LoadRequest(BaseModel):
    video_url: str

class ChatRequest(BaseModel):
    video_id: str
    question: str

# ──────────────────────────────────────────────
# Helpers
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

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ──────────────────────────────────────────────
# Build chain — lightweight version
# Uses HuggingFace Inference API for embeddings
# Uses FAISS instead of ChromaDB
# No local model downloads
# ──────────────────────────────────────────────
def build_chain(video_id: str):
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # 1. Fetch transcript
    api = YouTubeTranscriptApi()
    transcript_list = api.list_transcripts(video_id, cookies='cookies.txt').find_transcript(["en"]).fetch()
    transcript = " ".join([entry["text"] for entry in transcript_list])

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = splitter.create_documents([transcript])

    # 3. Embeddings via HuggingFace API (no local model download)
    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. FAISS vector store (much lighter than ChromaDB)
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding)
    retreiver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 5. Prompt (same as your notebook)
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
5. Answer in just 2 sentences maximum.

DETAILED RESPONSE:
"""
    prompt_1 = PromptTemplate(
        template=youtube_template,
        input_variables=["docs", "query"]
    )

    # 6. LLM via HuggingFace API (no local download)
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
    )
    model = ChatHuggingFace(llm=llm, temperature=0.7)

    # 7. Chain (same as your notebook)
    parallel_chain = RunnableParallel({
        "docs":  retreiver | RunnableLambda(format_docs),
        "query": RunnablePassthrough(),
    })
    parser = StrOutputParser()
    final_chain = parallel_chain | prompt_1 | model | parser

    return final_chain, transcript

# ──────────────────────────────────────────────
# Routes
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
# Serve frontend
# ──────────────────────────────────────────────
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
