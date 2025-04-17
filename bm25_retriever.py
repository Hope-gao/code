import streamlit as st
import pickle
import os
import re
import time
import numpy as np
import setup_nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # Keep if you chose Option B for NLTK setup
from rank_bm25 import BM25Okapi

# --- NLTK Setup ---
# (Same comment as in tfidf_retriever.py - better handled globally)
# try:
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
# except LookupError:
#     st.error("NLTK data missing in bm25_retriever.py. Ensure it's downloaded globally.")

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(__file__)
# 假设 'retriever_indices_chunked' 文件夹就在脚本所在的目录 (COMP5423 Project) 下
INDEX_SAVE_DIR = os.path.join(SCRIPT_DIR, 'retriever_indices')
# 构建索引文件的完整路径
BM25_INDEX_FILE = os.path.join(INDEX_SAVE_DIR, 'bm25_retriever_chunked.pkl')
# 打印路径以供调试（可以稍后移除）
print(f"[bm25_retriever.py] Calculated BM25 Index Path: {BM25_INDEX_FILE}")

# --- Preprocessing Functions ---
# Ensure these match the preprocessing used when building the index!
stop_words = set(stopwords.words('english')) # Define globally if not handled in app.py
lemmatizer = WordNetLemmatizer()             # Define globally if not handled in app.py

def preprocess_text_bm25(text):
    """Preprocesses text for BM25: remove HTML, non-alpha, lowercase, lemmatize, remove stopwords, return tokens."""
    if not isinstance(text, str): return []
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    # --- Use Lemmatization only if WordNet setup is confirmed working ---
    # filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    # --- Version without Lemmatization (if you chose Option A) ---
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

# --- BM25 Retriever Class (Adapted) ---
class Bm25Retriever:
    def __init__(self, bm25_index, chunk_ids, chunks_dict, original_doc_ids_map):
        self.bm25_index = bm25_index # This is the BM25Okapi object
        self.chunk_ids = chunk_ids
        self.chunks_dict = chunks_dict
        self.original_doc_ids_map = original_doc_ids_map
        self.using_chunks = True

    # !!! New Method: retrieve_passages - Standardized Interface !!!
    def retrieve_passages(self, query, top_k=5):
        """Retrieves top-k passages (chunks) with scores."""
        if self.bm25_index is None or not self.chunk_ids:
            st.error("BM25 retriever is not properly initialized.")
            return []
        if not query or not isinstance(query, str):
             st.warning("Invalid query provided to BM25 retriever.")
             return []

        tokenized_query = preprocess_text_bm25(query)
        if not tokenized_query:
            st.warning("Preprocessing removed the entire query.")
            return []

        try:
            # Get scores for all documents/chunks
            doc_scores = self.bm25_index.get_scores(tokenized_query)

            num_chunks = len(self.chunk_ids)
            actual_top_k = min(top_k, num_chunks)
            if actual_top_k <= 0: return []

            # Get top k indices using argsort
            # Ensure indices are within bounds
            top_k_indices = np.argsort(doc_scores)[::-1][:actual_top_k]

            results = []
            for i in top_k_indices:
                 # Ensure index is valid before accessing lists/dicts
                 if 0 <= i < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[i]
                    score = float(doc_scores[i]) # Ensure float
                    passage_text = self.chunks_dict.get(chunk_id, "Error: Chunk text not found")
                    original_doc_id = self.original_doc_ids_map.get(chunk_id, -1)

                    # BM25 scores can be <= 0, usually no need to filter low scores unless negative are problematic
                    results.append({
                        "doc_id": original_doc_id,
                        "passage_text": passage_text,
                        "score": score,
                        "passage_index_in_faiss": -1, # Not applicable for BM25
                        "chunk_id": chunk_id # Include chunk_id for reference
                    })
                 else:
                      st.warning(f"BM25 retrieved invalid index {i}. Skipping.")

            return results
        except Exception as e:
            st.error(f"Error during BM25 retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

# --- Loading Function ---
@st.cache_resource # Cache the loaded retriever object
def load_bm25_retriever(index_path=BM25_INDEX_FILE):
    """Loads the pickled BM25 retriever object."""
    if not os.path.exists(index_path):
        st.error(f"BM25 index file not found at: {index_path}")
        return None
    print(f"Loading BM25 retriever from: {index_path}")
    try:
        with open(index_path, 'rb') as f_in:
             # Load the dictionary or object saved during training
            saved_data = pickle.load(f_in)

             # --- Adapt based on how you saved the .pkl file ---
             # Option 1: Saved the entire Bm25Retriever CLASS instance
            if isinstance(saved_data, Bm25Retriever):
                 retriever_obj = saved_data
             # Option 2: Saved a dictionary containing the necessary components
            elif isinstance(saved_data, dict):
                 retriever_obj = Bm25Retriever(
                     bm25_index=saved_data.get('bm25_index'), # BM25Okapi object
                     chunk_ids=saved_data.get('chunk_ids'),
                     chunks_dict=saved_data.get('chunks_dict'),
                     original_doc_ids_map=saved_data.get('original_doc_ids_map')
                 )
            else:
                 st.error(f"Loaded BM25 pkl file has unexpected format: {type(saved_data)}")
                 return None

            # --- Validation ---
            if not all([retriever_obj.bm25_index, retriever_obj.chunk_ids,
                         retriever_obj.chunks_dict, retriever_obj.original_doc_ids_map]):
                st.error("Loaded BM25 retriever object is missing required components.")
                return None

            print("BM25 retriever loaded successfully.")
            return retriever_obj

    except (pickle.UnpicklingError, EOFError) as pe:
        st.error(f"Error unpickling BM25 index file (it might be corrupted or incompatible): {pe}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading BM25 retriever: {e}")
        import traceback
        traceback.print_exc()
        return None