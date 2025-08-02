# rag_engine.py

import os
import json
from PyPDF2 import PdfReader
from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext, load_index_from_storage
# from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re

# This will use your GPU if torch detects it!
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# load_dotenv()

# def parse_pdfs(doc_folder="docs"):
#     """
#     Parses each PDF file in the doc_folder, returning a list of dicts:
#     [
#       {
#         "file_name": ...,
#         "page_num": ...,
#         "text": ...,
#       },
#       ...
#     ]
#     """
#     parsed_pages = []
#     for file in os.listdir(doc_folder):
#         if file.lower().endswith(".pdf"):
#             pdf_path = os.path.join(doc_folder, file)
#             reader = PdfReader(pdf_path)
#             for i, page in enumerate(reader.pages):
#                 text = page.extract_text() or ""
#                 parsed_pages.append({
#                     "file_name": file,
#                     "page_num": i + 1,  # 1-based index for humans
#                     "text": text.strip(),
#                 })
#     return parsed_pages



def parse_pdfs(doc_folder="docs"):
    parsed_chunks = []
    for file in os.listdir(doc_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(doc_folder, file)
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                # Split by sentences using regex (handles ., !, ? end punctuation)
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for j, sent in enumerate(sentences):
                    sent = sent.strip()
                    if sent:
                        parsed_chunks.append({
                            "file_name": file,
                            "page_num": i + 1,
                            "chunk_num": j + 1,
                            "text": sent,
                        })
    return parsed_chunks



def save_parsed_chunks(parsed_chunks, out_path="parsed_chunks.json"):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed_chunks, f, ensure_ascii=False, indent=2)

# rag_engine.py (add below your current code)



def build_index(parsed_chunks, persist_dir="storage"):
    """
    Indexes parsed pages with LlamaIndex + Chroma, persists index to disk.
    Each page becomes a Document with metadata for citation.
    """
    # Prepare LlamaIndex Documents
    docs = [
        Document(
            text=chunk['text'],
            metadata={
                "file_name": chunk['file_name'],
                "page_num": chunk['page_num'],
                "chunk_num": chunk['chunk_num'],
            }
        )
        for chunk in parsed_chunks if chunk['text']
    ]

    # Setup Chroma as persistent vector DB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("rag_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Build and persist index
    index = VectorStoreIndex.from_documents(
        docs,
        vector_store=vector_store,
        embed_model=embed_model,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir)
    return index  # You may return for further use (optional)



def load_index(persist_dir="storage"):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    return index



# # Test the function standalone
# if __name__ == "__main__":
#     pages = parse_pdfs("docs")
#     for page in pages:
#         print(f"{page['file_name']} | Page {page['page_num']}: {page['text'][:100]}...")
#     save_parsed_chunks(pages)
#     build_index(pages)  # <-- Fixed here!
#     index = load_index()
#     engine = index.as_query_engine()
#     print(engine.query("What is this document about?"))
