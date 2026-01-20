import pandas as pd
import numpy as np
import uuid
import chromadb
from pathlib import Path
from tqdm.auto import tqdm

from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. SETUP & CONFIGURATION
ROOT = Path(".").resolve() # Adjusted to current directory for standard script use
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "filtered_complaints.csv"
VECTOR_STORE_DIR = ROOT / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 10000
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "complaint_embeddings"

def create_chunks(df, text_splitter, text_col='cleaned_narrative'):
    """Iterates through rows and splits long text into smaller chunks."""
    chunks = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue
            
        texts = text_splitter.split_text(text)
        
        for i, chunk_text in enumerate(texts):
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'complaint_id': str(row.get('Complaint ID', 'unknown')),
                'product': row['Product'],
                'text': chunk_text,
                'chunk_index': i,
                'original_index': idx
            })
    return pd.DataFrame(chunks)

def main():
    # --- STEP 1: LOAD & SAMPLE ---
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: File not found at {PROCESSED_DATA_PATH}")
        return

    print(f"Loading data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Total records in CSV: {len(df)}")

    print("Performing stratified sampling...")
    df_sample = df.groupby('Product', group_keys=False).apply(
        lambda x: x.sample(frac=min(SAMPLE_SIZE/len(df), 1.0), random_state=42)
    ).reset_index(drop=True)
    print(f"Sampled records: {len(df_sample)}")

    # --- STEP 2: CHUNKING ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    df_chunks = create_chunks(df_sample, text_splitter)
    print(f"Created {len(df_chunks)} chunks.")

    # --- STEP 3: EMBEDDING ---
    print(f"Loading embedding model: {MODEL_NAME}")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    texts = df_chunks['text'].tolist()
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)

    # --- STEP 4: INDEXING IN CHROMADB ---
    print(f"Initializing ChromaDB at: {VECTOR_STORE_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    ids = df_chunks['chunk_id'].tolist()
    metadatas = df_chunks[['complaint_id', 'product', 'chunk_index']].to_dict(orient='records')
    documents = df_chunks['text'].tolist()

    # Batch insert to handle large datasets
    db_batch_size = 5000
    for i in tqdm(range(0, len(ids), db_batch_size), desc="Indexing to ChromaDB"):
        end_idx = min(i + db_batch_size, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            metadatas=metadatas[i:end_idx],
            documents=documents[i:end_idx]
        )

    print(f"\nSUCCESS: Indexed {collection.count()} chunks in collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()