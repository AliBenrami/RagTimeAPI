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
import re
import json
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
    chunks = chunk_pdf(pdf_content)
    
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

def chunk_pdf(pdf_content):
    # Split by paragraph breaks (double newlines or more)
    paragraphs = re.split(r'\n\s*\n+', pdf_content.strip())
    # Filter out empty paragraphs and clean up
    chunks = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    return chunks

def extract_ratings_from_chunks(doc_chunks):
    """
    Extract ratings using LLM analysis
    """
    if not doc_chunks:
        return []
    
    # Combine chunks for analysis
    combined_text = " ".join(doc_chunks)
    
    rating_prompt = f"""
    Analyze the following text and extract any ratings, scores, or evaluations mentioned.
    Look for ratings on scales like 1-5, 1-10, 0-100, percentages, or star ratings.
    
    Text: {combined_text}
    
    Return your response as a JSON array with objects containing:
    - "rating": the numerical rating (normalized to 0-5 scale)
    - "source_text": the specific text where the rating was found
    - "confidence": your confidence level (0-1)
    
    If no ratings are found, return an empty array [].
    
    Response (JSON only):
    """
    
    try:
        response = call_llm(rating_prompt, [], "gemini-2.0-flash", "")
        # Try to parse JSON response
        ratings = json.loads(response)
        return ratings if isinstance(ratings, list) else []
    except Exception as e:
        print(f"Error extracting ratings with LLM: {e}")
        return []

def get_confidence_rating(llm_response: str):
    """
    Extract confidence rating from format [Confidence Rating: X/5]
    Returns the rating as a string, or None if not found
    """
    confidence_regex = r"\[Confidence Rating:\s*(\d+(?:\.\d+)?)/5\]"
    match = re.search(confidence_regex, llm_response)
    if match:
        return match.group(1) + "/5"
    return None

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
    doc_chunks_result = get_sentence_similarity(request.prompt)
    doc_chunks = doc_chunks_result[0] if doc_chunks_result and len(doc_chunks_result) > 0 else []
    
    system_instruction = """
    You are a friendly and helpful AI assistant. Answer the user's question directly and naturally.

    **Your task:** Answer the user's question using the information provided below. If the information doesn't help answer their question, just answer based on your general knowledge.

    **Critical instructions:**
    - Answer the user's question directly - don't talk about the information provided
    - Don't mention "context", "information provided", or "based on the documents"
    - Don't reference or quote the provided information
    - Just give a natural, conversational answer as if you're talking to a friend
    - If it's a greeting, respond with a friendly greeting back
    
    **Confidence rating:** At the end of your response, rate how confident you are in your answer:
    - 5/5: The provided information directly and completely answers the question
    - 4/5: The information provides most but not all relevant details
    - 3/5: The information is somewhat helpful but has significant gaps
    - 2/5: The information is only tangentially related
    - 1/5: The information is not relevant or helpful
    
    End with: [Confidence Rating: X/5]
    """
    
    # Format the document chunks as a readable string
    doc_chunks_text = "\n".join(doc_chunks) if doc_chunks else "No relevant information found."
    
    user_prompt = f"""
    this is the information: {doc_chunks_text}
    this is the user prompt: {request.prompt}
    """
    
    response = call_llm(user_prompt, request.history, model, system_instruction)
    confidence_rating = get_confidence_rating(response)
    
    #remove the confidence rating from the response
    response = response.split("[Confidence Rating:")[0]
    
    
    return {
        "response": response, 
        "doc_chunks": doc_chunks,
        "confidence_rating": confidence_rating
    }
