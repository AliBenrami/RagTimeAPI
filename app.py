from fastapi import FastAPI, File, UploadFile
import os
import shutil
import hashlib
import os
from llm import call_llm
import chromadb
from sentence_transformers import SentenceTransformer
import fitz
from pydantic import BaseModel
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware

chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class RagRequest(BaseModel):
    prompt: str
    history: list = []


def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def get_sentence_similarity(query, top_k=20):
    doc_embeddings = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents"]
    )["documents"]

    return doc_embeddings

def does_pdf_exist(pdf_name):
    file_path = os.path.join("uploads", pdf_name)
    return os.path.isfile(file_path)

def get_content_hash(content):
    """Generate a hash of the content to detect duplicates"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def does_content_exist_in_chroma(content_hash):
    """Check if content with this hash already exists in Chroma"""
    try:
        # Query Chroma for documents with this content hash
        results = collection.query(
            query_texts=[""],  # Empty query to get all documents
            n_results=1000,  # Adjust based on your expected collection size
            where={"content_hash": content_hash},
            include=["metadatas"]
        )
        return len(results["metadatas"][0]) > 0
    except Exception as e:
        print(f"Error checking Chroma for existing content: {e}")
        return False

def upload_to_chroma(pdf_content, pdf_name):
    # Generate content hash for duplicate detection
    content_hash = get_content_hash(pdf_content)
    
    # Check if content already exists in Chroma
    if does_content_exist_in_chroma(content_hash):
        return {"status": "duplicate", "message": "Content already exists in database"}
    
    #note to self fix chunking so that it doesn't split in the middle of a sentence
    #chunk the pdf content 
    chunk_size = 1000
    chunks = [pdf_content[i:i+chunk_size] for i in range(0, len(pdf_content), chunk_size)]
    
    #embed the chunks
    embeddings = bi_encoder.encode(chunks, convert_to_tensor=True).tolist()
    
    #add the chunks to the chroma collection with content hash
    collection.add(
        ids=[str(i) + "_" + pdf_name + "_" + str(content_hash) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{
            "pdf_name": pdf_name, 
            "chunk_index": i,
            "content_hash": content_hash
        } for i in range(len(chunks))]
    )
    
    return {"status": "success", "chunks_added": len(chunks)}


# Upload endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Check if file already exists on disk
    if does_pdf_exist(file.filename):
        return {"error": "File already exists on disk", "filename": file.filename}
    
    temp_path = f"uploads/{file.filename}"
    
    try:
        # Save the uploaded PDF to disk
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Read PDF content
        pdf_content = read_pdf(temp_path)
        
        if not pdf_content or len(pdf_content.strip()) == 0:
            # Remove empty file
            os.remove(temp_path)
            return {"error": "PDF appears to be empty or could not be read"}
        
        # Upload to Chroma with duplicate detection
        upload_result = upload_to_chroma(pdf_content, file.filename)
        
        # Remove the file because it would be expensive to store a bunck of pdfs locally and on db 
        os.remove(temp_path)
        
        if upload_result["status"] == "duplicate":
            # Remove the file since content already exists
            return {"warning": upload_result["message"], "filename": file.filename}
        
        return {
            "success": True, 
            "filename": file.filename,
            "chunks_added": upload_result["chunks_added"]
        }
        
    except Exception as e:
        # Clean up file if there was an error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"error": f"Failed to process PDF: {str(e)}"}

@app.post("/queryall")
async def queryall():
    #query vector db
    doc_chunks = collection.get()
    return {"doc_chunks": doc_chunks}

@app.post("/query/{query}")
async def query(query: str, top_k: int = 20):
    #query vector db
    doc_chunks = get_sentence_similarity(query, top_k)
    return {"query": query, "doc_chunks": doc_chunks}

@app.post("/call-rag-llm")
async def call_rag_llm(request: RagRequest = Body(...), model: str = "gemini-2.0-flash"):
    context_prompt = """Use the following context to inform your responses. 
    Do not repeat the context back to me unless I ask for it. 
    If the context answers the question directly, use it confidently. 
    If the context partially applies, combine it with your own knowledge to give a complete answer. 
    If the context contradicts known facts, mention the conflict. Be concise, accurate, and context-aware at all times.
    give a rating of 1 to 10 for the relevance of the context to the question.
    """
    doc_chunks = get_sentence_similarity(request.prompt)
    response = call_llm(request.prompt + "\n" + f"{context_prompt}\n" + "\n".join(doc_chunks[0]), request.history, model)
    #get ratings as a separate llm call or use a regx to extract int


    return {"response": response, "doc_chunks": doc_chunks}
    





