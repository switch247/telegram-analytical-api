import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ChromaDB imports
try:
    import chromadb
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError:
    chromadb = None

class RAGSystem:
    def __init__(self, 
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 llm_model_name: str = 'google/flan-t5-base',
                 parquet_path: str = None,
                 chromadb_dir: str = None,
                 chromadb_collection: str = 'complaint_embeddings',
                 top_k: int = 5):
        """
        Initializes the RAG system by loading data and setting up the search index.
        Supports Parquet/FAISS or ChromaDB retrieval.
        If parquet_path is provided, uses FAISS. If chromadb_dir is provided, uses ChromaDB.
        """
        self.mode = None
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.generator = pipeline("text2text-generation", model=llm_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        if parquet_path is not None:
            print("[RAG] Using Parquet/FAISS mode.")
            self.mode = 'parquet'
            self.df = pd.read_parquet(parquet_path)
            embeddings = np.array(self.df['embeddings'].tolist()).astype('float32')
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        elif chromadb_dir is not None and chromadb is not None:
            print("[RAG] Using ChromaDB mode.")
            self.mode = 'chroma'
            embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
            self.vector_store = Chroma(
                client=chromadb.PersistentClient(path=chromadb_dir),
                collection_name=chromadb_collection,
                embedding_function=embedding_function
            )
        else:
            raise ValueError("Must provide either parquet_path or chromadb_dir (with ChromaDB installed)")

    def retrieve(self, question: str, k: int = None):
        """
        Finds the most relevant complaints for a question.
        """
        k = k or self.top_k
        if self.mode == 'parquet':
            question_enc = self.embedding_model.encode([question]).astype('float32')
            distances, indices = self.index.search(question_enc, k)
            results = self.df.iloc[indices[0]]
            return results['complaint_text'].tolist()
        elif self.mode == 'chroma':
            docs = self.vector_store.similarity_search(question, k=k)
            return [doc.page_content for doc in docs]
        else:
            raise RuntimeError("RAGSystem not properly initialized.")

    def augment_and_generate(self, question: str, context_chunks):
        """
        Combines chunks into a prompt and asks the LLM.
        """
        context_str = "\n\n".join([f"Excerpt {i+1}: {text}" for i, text in enumerate(context_chunks)])
        prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use ONLY the provided context to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context_str}

Question: {question}

Answer:"""
        response = self.generator(prompt, max_length=512, truncation=True)
        return response[0]['generated_text']

    def ask(self, question: str, k: int = None):
        """
        High-level function to run the full RAG pipeline.
        """
        k = k or self.top_k
        context_chunks = self.retrieve(question, k=k)
        answer = self.augment_and_generate(question, context_chunks)
        return {
            "question": question,
            "answer": answer,
            "sources": context_chunks[:2]
        }
