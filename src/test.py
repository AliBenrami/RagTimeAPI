"""
Comprehensive Test Suite for RagTimeAPI

This test file contains comprehensive tests for all functions in src/app.py:

FUNCTION TESTS:
1. get_confidence_rating() - Tests confidence rating extraction from LLM responses
2. get_content_hash() - Tests MD5 hash generation for content deduplication
3. does_pdf_exist() - Tests PDF file existence checking
4. chunk_pdf() - Tests PDF content chunking by sentences
5. does_content_exist_in_chroma() - Tests ChromaDB duplicate content detection
6. upload_to_chroma() - Tests PDF content upload to ChromaDB
7. get_sentence_similarity() - Tests semantic similarity search
8. read_pdf() - Tests PDF text extraction

FASTAPI ENDPOINT TESTS:
1. /upload - Tests PDF upload endpoint with various scenarios
2. /query/{query} - Tests query endpoint for semantic search
3. /queryall - Tests endpoint to get all documents
4. /call-rag-llm - Tests RAG LLM endpoint with confidence rating

EDGE CASES & ERROR HANDLING:
- Empty content handling
- Unicode content handling
- File system edge cases
- Database error handling
- Invalid input handling
- Orphaned code detection

MOCKING:
- All external dependencies are mocked (ChromaDB, SentenceTransformer, fitz)
- FastAPI TestClient for endpoint testing
- File system operations mocked where appropriate

INTEGRATION TESTS:
- Marked with @pytest.mark.integration for tests requiring actual dependencies
"""

import src.app as app
import pytest
import os
import tempfile
import hashlib
from unittest.mock import Mock, patch, MagicMock
import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# Test get_confidence_rating function
def test_get_confidence_rating():
    response = app.get_confidence_rating("Hello, how are you? [Confidence Rating: 4/5]")
    assert response == "4/5"


def test_get_confidence_rating_with_decimal():
    response = app.get_confidence_rating("This is a test response. [Confidence Rating: 3.5/5]")
    assert response == "3.5/5"


def test_get_confidence_rating_no_rating():
    response = app.get_confidence_rating("This response has no confidence rating.")
    assert response is None


def test_get_confidence_rating_invalid_format():
    response = app.get_confidence_rating("This has [Confidence Rating: invalid] format")
    assert response is None


# Test get_content_hash function
def test_get_content_hash():
    content = "This is test content"
    expected_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    result = app.get_content_hash(content)
    assert result == expected_hash


def test_get_content_hash_empty_string():
    content = ""
    expected_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    result = app.get_content_hash(content)
    assert result == expected_hash


def test_get_content_hash_unicode():
    content = "Test content with émojis 🚀 and unicode"
    expected_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    result = app.get_content_hash(content)
    assert result == expected_hash


# Test does_pdf_exist function
def test_does_pdf_exist_file_exists():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(dir="uploads", suffix=".pdf", delete=False) as temp_file:
        temp_filename = os.path.basename(temp_file.name)
    
    try:
        result = app.does_pdf_exist(temp_filename)
        assert result is True
    finally:
        # Clean up
        os.remove(os.path.join("uploads", temp_filename))


def test_does_pdf_exist_file_not_exists():
    result = app.does_pdf_exist("nonexistent_file.pdf")
    assert result is False


# Test chunk_pdf function
def test_chunk_pdf_simple_sentences():
    content = "This is sentence one. This is sentence two. This is sentence three!"
    chunks = app.chunk_pdf(content)
    expected = ["This is sentence one.", "This is sentence two.", "This is sentence three!"]
    assert chunks == expected


def test_chunk_pdf_with_question_marks():
    content = "What is this? This is a question. And this is a statement."
    chunks = app.chunk_pdf(content)
    expected = ["What is this?", "This is a question.", "And this is a statement."]
    assert chunks == expected


def test_chunk_pdf_with_exclamation_marks():
    content = "Hello! How are you? I'm doing great!"
    chunks = app.chunk_pdf(content)
    expected = ["Hello!", "How are you?", "I'm doing great!"]
    assert chunks == expected


def test_chunk_pdf_empty_content():
    content = ""
    chunks = app.chunk_pdf(content)
    assert chunks == []


def test_chunk_pdf_whitespace_only():
    content = "   \n\t   "
    chunks = app.chunk_pdf(content)
    assert chunks == []


def test_chunk_pdf_single_sentence():
    content = "This is a single sentence."
    chunks = app.chunk_pdf(content)
    assert chunks == ["This is a single sentence."]


def test_chunk_pdf_multiple_spaces():
    content = "Sentence one.    Sentence two.   Sentence three."
    chunks = app.chunk_pdf(content)
    expected = ["Sentence one.", "Sentence two.", "Sentence three."]
    assert chunks == expected


# Test does_content_exist_in_chroma function
@patch('src.app.collection')
def test_does_content_exist_in_chroma_content_exists(mock_collection):
    # Mock the collection query to return existing content
    mock_collection.query.return_value = {
        "metadatas": [{"content_hash": "test_hash", "pdf_name": "test.pdf"}]
    }
    
    result = app.does_content_exist_in_chroma("test_hash")
    assert result is True


@patch('src.app.collection')
def test_does_content_exist_in_chroma_content_not_exists(mock_collection):
    # Mock the collection query to return no content
    mock_collection.query.return_value = {
        "metadatas": [[]]
    }
    
    result = app.does_content_exist_in_chroma("test_hash")
    assert result is False


@patch('src.app.collection')
def test_does_content_exist_in_chroma_exception(mock_collection):
    # Mock the collection query to raise an exception
    mock_collection.query.side_effect = Exception("Database error")
    
    result = app.does_content_exist_in_chroma("test_hash")
    assert result is False


# Test upload_to_chroma function
@patch('src.app.does_content_exist_in_chroma')
@patch('src.app.chunk_pdf')
@patch('src.app.bi_encoder')
@patch('src.app.collection')
def test_upload_to_chroma_duplicate_content(mock_collection, mock_encoder, mock_chunk, mock_exists):
    # Mock duplicate content detection
    mock_exists.return_value = True
    
    result = app.upload_to_chroma("test content", "test.pdf")
    
    assert result["status"] == "duplicate"
    assert "Content already exists in database" in result["message"]
    # Ensure no chunks were processed
    mock_chunk.assert_not_called()
    mock_encoder.encode.assert_not_called()
    mock_collection.add.assert_not_called()


@patch('src.app.does_content_exist_in_chroma')
@patch('src.app.chunk_pdf')
@patch('src.app.bi_encoder')
@patch('src.app.collection')
def test_upload_to_chroma_success(mock_collection, mock_encoder, mock_chunk, mock_exists):
    # Mock new content
    mock_exists.return_value = False
    mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
    mock_encoder.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    result = app.upload_to_chroma("test content", "test.pdf")
    
    assert result["status"] == "success"
    assert result["chunks_added"] == 2
    mock_chunk.assert_called_once_with("test content")
    mock_encoder.encode.assert_called_once()
    mock_collection.add.assert_called_once()


# Test get_sentence_similarity function
@patch('src.app.collection')
def test_get_sentence_similarity(mock_collection):
    # Mock the collection query
    mock_collection.query.return_value = {
        "documents": [["doc1", "doc2", "doc3"]]
    }
    
    result = app.get_sentence_similarity("test query", top_k=3)
    
    assert result == [["doc1", "doc2", "doc3"]]
    mock_collection.query.assert_called_once_with(
        query_texts=["test query"],
        n_results=3,
        include=["documents"]
    )


@patch('src.app.collection')
def test_get_sentence_similarity_default_top_k(mock_collection):
    # Mock the collection query
    mock_collection.query.return_value = {
        "documents": [["doc1", "doc2"]]
    }
    
    result = app.get_sentence_similarity("test query")
    
    assert result == [["doc1", "doc2"]]
    mock_collection.query.assert_called_once_with(
        query_texts=["test query"],
        n_results=20,  # Default value
        include=["documents"]
    )


# Test read_pdf function
@patch('fitz.open')
def test_read_pdf_success(mock_fitz_open):
    # Mock the PDF document
    mock_doc = Mock()
    mock_page1 = Mock()
    mock_page1.get_text.return_value = "Page 1 content"
    mock_page2 = Mock()
    mock_page2.get_text.return_value = "Page 2 content"
    mock_doc.__iter__.return_value = [mock_page1, mock_page2]
    mock_fitz_open.return_value = mock_doc
    
    result = app.read_pdf("test.pdf")
    
    expected = "Page 1 content\nPage 2 content\n"
    assert result == expected
    mock_fitz_open.assert_called_once_with("test.pdf")


@patch('fitz.open')
def test_read_pdf_empty_document(mock_fitz_open):
    # Mock empty PDF document
    mock_doc = Mock()
    mock_doc.__iter__.return_value = []
    mock_fitz_open.return_value = mock_doc
    
    result = app.read_pdf("empty.pdf")
    
    assert result == ""


@patch('fitz.open')
def test_read_pdf_single_page(mock_fitz_open):
    # Mock single page PDF
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Single page content"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc
    
    result = app.read_pdf("single.pdf")
    
    assert result == "Single page content\n"


# Test edge cases and error handling
def test_get_confidence_rating_edge_cases():
    # Test with different formats
    assert app.get_confidence_rating("[Confidence Rating: 1/5]") == "1/5"
    assert app.get_confidence_rating("[Confidence Rating: 5/5]") == "5/5"
    assert app.get_confidence_rating("[Confidence Rating: 0/5]") == "0/5"
    assert app.get_confidence_rating("[Confidence Rating: 2.75/5]") == "2.75/5"


def test_chunk_pdf_edge_cases():
    # Test with no sentence endings
    content = "This is a sentence without ending punctuation"
    chunks = app.chunk_pdf(content)
    assert chunks == ["This is a sentence without ending punctuation"]
    
    # Test with multiple punctuation marks
    content = "Sentence one!!! Sentence two??? Sentence three..."
    chunks = app.chunk_pdf(content)
    expected = ["Sentence one!!!", "Sentence two???", "Sentence three..."]
    assert chunks == expected


def test_get_content_hash_consistency():
    # Test that same content always produces same hash
    content = "Test content"
    hash1 = app.get_content_hash(content)
    hash2 = app.get_content_hash(content)
    assert hash1 == hash2
    
    # Test that different content produces different hashes
    content2 = "Different test content"
    hash3 = app.get_content_hash(content2)
    assert hash1 != hash3


# Test FastAPI endpoints
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json

client = TestClient(app.app)


@patch('src.app.does_pdf_exist')
@patch('src.app.read_pdf')
@patch('src.app.upload_to_chroma')
def test_upload_pdf_endpoint_success(mock_upload, mock_read, mock_exists):
    # Mock dependencies
    mock_exists.return_value = False
    mock_read.return_value = "PDF content"
    mock_upload.return_value = {"status": "success", "chunks_added": 5}
    
    # Create a mock file
    files = {"file": ("test.pdf", b"fake pdf content", "application/pdf")}
    
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "test.pdf"
    assert data["chunks_added"] == 5


@patch('src.app.does_pdf_exist')
def test_upload_pdf_endpoint_file_exists(mock_exists):
    mock_exists.return_value = True
    
    files = {"file": ("existing.pdf", b"fake pdf content", "application/pdf")}
    
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "File already exists on disk" in data["error"]


@patch('src.app.does_pdf_exist')
@patch('src.app.read_pdf')
@patch('src.app.upload_to_chroma')
def test_upload_pdf_endpoint_duplicate_content(mock_upload, mock_read, mock_exists):
    mock_exists.return_value = False
    mock_read.return_value = "PDF content"
    mock_upload.return_value = {"status": "duplicate", "message": "Content already exists in database"}
    
    files = {"file": ("test.pdf", b"fake pdf content", "application/pdf")}
    
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "warning" in data
    assert "Content already exists in database" in data["warning"]


@patch('src.app.get_sentence_similarity')
def test_query_endpoint(mock_similarity):
    mock_similarity.return_value = [["doc1", "doc2"]]
    
    response = client.post("/query/test%20query?top_k=5")
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["doc_chunks"] == [["doc1", "doc2"]]


@patch('src.app.collection')
def test_queryall_endpoint(mock_collection):
    mock_collection.get.return_value = {"documents": ["doc1", "doc2"]}
    
    response = client.post("/queryall")
    
    assert response.status_code == 200
    data = response.json()
    assert "doc_chunks" in data


@patch('src.app.get_sentence_similarity')
@patch('src.app.call_llm')
def test_call_rag_llm_endpoint(mock_call_llm, mock_similarity):
    mock_similarity.return_value = [["context doc1", "context doc2"]]
    mock_call_llm.return_value = "This is a helpful response. [Confidence Rating: 4/5]"
    
    request_data = {
        "prompt": "What is the main topic?",
        "history": []
    }
    
    response = client.post("/call-rag-llm", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "doc_chunks" in data
    assert "confidence_rating" in data
    assert data["confidence_rating"] == "4/5"


# Test error handling in chunk_pdf function
def test_chunk_pdf_with_orphaned_code():
    """
    Test that the chunk_pdf function works correctly despite the orphaned code
    that appears to be from a different function in the source.
    """
    content = "This is a test. It has multiple sentences. Each should be a chunk!"
    chunks = app.chunk_pdf(content)
    expected = ["This is a test.", "It has multiple sentences.", "Each should be a chunk!"]
    assert chunks == expected


def test_chunk_pdf_orphaned_code_unreachable():
    """
    Test that the orphaned code in chunk_pdf function is indeed unreachable.
    The function should return before reaching the orphaned rating extraction code.
    """
    content = "Test content."
    chunks = app.chunk_pdf(content)
    
    # The function should return the chunks and not execute the orphaned code
    # that references undefined variables like 'doc_chunks'
    assert chunks == ["Test content."]
    # If the orphaned code were reachable, it would cause a NameError
    # because 'doc_chunks' is not defined in the function scope


# Test extract_ratings_from_chunks function (newly extracted from orphaned code)
@patch('src.app.call_llm')
def test_extract_ratings_from_chunks_success(mock_call_llm):
    # Mock successful LLM response
    mock_call_llm.return_value = '[{"rating": 4.5, "source_text": "This is great", "confidence": 0.9}]'
    
    doc_chunks = ["This is great content.", "It has a rating of 4.5."]
    result = app.extract_ratings_from_chunks(doc_chunks)
    
    assert len(result) == 1
    assert result[0]["rating"] == 4.5
    assert result[0]["source_text"] == "This is great"
    assert result[0]["confidence"] == 0.9


@patch('src.app.call_llm')
def test_extract_ratings_from_chunks_no_ratings(mock_call_llm):
    # Mock response with no ratings
    mock_call_llm.return_value = '[]'
    
    doc_chunks = ["This is regular content.", "No ratings here."]
    result = app.extract_ratings_from_chunks(doc_chunks)
    
    assert result == []


@patch('src.app.call_llm')
def test_extract_ratings_from_chunks_empty_chunks(mock_call_llm):
    result = app.extract_ratings_from_chunks([])
    
    assert result == []
    mock_call_llm.assert_not_called()


@patch('src.app.call_llm')
def test_extract_ratings_from_chunks_llm_error(mock_call_llm):
    # Mock LLM error
    mock_call_llm.side_effect = Exception("LLM API error")
    
    doc_chunks = ["Test content."]
    result = app.extract_ratings_from_chunks(doc_chunks)
    
    assert result == []


@patch('src.app.call_llm')
def test_extract_ratings_from_chunks_invalid_json(mock_call_llm):
    # Mock invalid JSON response
    mock_call_llm.return_value = "Invalid JSON response"
    
    doc_chunks = ["Test content."]
    result = app.extract_ratings_from_chunks(doc_chunks)
    
    assert result == []


# Test edge cases for confidence rating extraction
def test_get_confidence_rating_edge_cases_extended():
    # Test with extra whitespace
    assert app.get_confidence_rating("Response [Confidence Rating: 3/5]") == "3/5"
    assert app.get_confidence_rating("Response [Confidence Rating:  4/5  ]") == "4/5"
    
    # Test with different number formats
    assert app.get_confidence_rating("Response [Confidence Rating: 0.5/5]") == "0.5/5"
    assert app.get_confidence_rating("Response [Confidence Rating: 4.0/5]") == "4.0/5"
    
    # Test with no rating
    assert app.get_confidence_rating("Just a regular response") is None
    assert app.get_confidence_rating("") is None


# Test content hash edge cases
def test_get_content_hash_edge_cases():
    # Test with very long content
    long_content = "A" * 10000
    hash1 = app.get_content_hash(long_content)
    hash2 = app.get_content_hash(long_content)
    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 hash length
    
    # Test with special characters
    special_content = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
    hash_special = app.get_content_hash(special_content)
    assert len(hash_special) == 32


# Test PDF existence function edge cases
def test_does_pdf_exist_edge_cases():
    # Test with None filename
    assert app.does_pdf_exist(None) is False
    
    # Test with empty filename
    assert app.does_pdf_exist("") is False
    
    # Test with path traversal attempt
    assert app.does_pdf_exist("../../../etc/passwd") is False


# Integration test setup (requires actual dependencies)
@pytest.mark.integration
def test_full_workflow():
    """
    Integration test that tests the full workflow.
    This test requires actual ChromaDB and SentenceTransformer dependencies.
    """
    # This would be a more complex test that tests the actual integration
    # between components. Marked as integration test to be run separately.
    pass


if __name__ == "__main__":
    pytest.main([__file__])






