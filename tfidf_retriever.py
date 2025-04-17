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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Setup (Only if not handled globally in app.py, prefer global setup) ---
# It's better to handle NLTK downloads ONCE in the main app.py
# If you put it here, ensure paths are correct relative to this file or use absolute paths.
# try:
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     # Make sure punkt, stopwords, wordnet, omw-1.4 are downloaded
# except LookupError:
#     st.error("NLTK data missing in tfidf_retriever.py. Ensure it's downloaded globally.")
#     # Handle error appropriately, maybe stop the app

# --- Configuration ---
# Make this path configurable or relative to the main app
# Example: Assuming indices are in a specific subfolder
SCRIPT_DIR = os.path.dirname(__file__)
INDEX_SAVE_DIR = os.path.join(SCRIPT_DIR, 'retriever_indices')
TFIDF_INDEX_FILE = os.path.join(INDEX_SAVE_DIR, 'tfidf_retriever_chunked.pkl')
print(f"[tfidf_retriever.py] Calculated TFIDF Index Path: {TFIDF_INDEX_FILE}")

# --- Preprocessing Functions (Copied from your training script) ---
# Ensure these match the preprocessing used when building the index!
stop_words = set(stopwords.words('english')) # Define globally if not handled in app.py
lemmatizer = WordNetLemmatizer()             # Define globally if not handled in app.py

def preprocess_text_tfidf(text):
    """Preprocesses text for TF-IDF: remove HTML, non-alpha, lowercase, lemmatize, remove stopwords."""
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    # --- Use Lemmatization only if WordNet setup is confirmed working ---
    # filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    # --- Version without Lemmatization (if you chose Option A) ---
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

# --- TF-IDF Retriever Class (Adapted) ---
class TfidfRetriever:
    def __init__(self, vectorizer, tfidf_matrix, chunk_ids, chunks_dict, original_doc_ids_map):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.chunk_ids = chunk_ids          # List of chunk_ids
        self.chunks_dict = chunks_dict      # Dict: chunk_id -> chunk_text
        self.original_doc_ids_map = original_doc_ids_map # Dict: chunk_id -> original_document_id
        self.using_chunks = True

    # !!! New Method: retrieve_passages - Standardized Interface !!!
    def retrieve_passages(self, query, top_k=5):
        """Retrieves top-k passages (chunks) with scores."""
        if self.tfidf_matrix is None or not self.chunk_ids:
            st.error("TF-IDF retriever is not properly initialized.")
            return []
        if not query or not isinstance(query, str):
            st.warning("Invalid query provided to TF-IDF retriever.")
            return []

        processed_query = preprocess_text_tfidf(query)
        if not processed_query:
            st.warning("Preprocessing removed the entire query.")
            return []

        try:
            query_vector = self.vectorizer.transform([processed_query])
            # Use densify() for easier score extraction if matrix is sparse
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            cosine_similarities = np.nan_to_num(cosine_similarities)

            num_chunks = len(self.chunk_ids)
            actual_top_k = min(top_k, num_chunks)
            if actual_top_k <= 0: return []

            # Get top k indices using argsort
            # Ensure indices are within bounds (should be by construction)
            top_k_indices = np.argsort(cosine_similarities)[::-1][:actual_top_k]

            results = []
            for i in top_k_indices:
                # Ensure index is valid before accessing lists/dicts
                if 0 <= i < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[i]
                    score = float(cosine_similarities[i]) # Ensure float
                    passage_text = self.chunks_dict.get(chunk_id, "Error: Chunk text not found")
                    original_doc_id = self.original_doc_ids_map.get(chunk_id, -1) # Use -1 for missing ID

                    # Filter very low scores if desired
                    if score > 1e-6:
                        results.append({
                            "doc_id": original_doc_id,
                            "passage_text": passage_text,
                            "score": score,
                            "passage_index_in_faiss": -1, # Not applicable for TF-IDF
                            "chunk_id": chunk_id # Include chunk_id for reference
                        })
                else:
                     st.warning(f"TF-IDF retrieved invalid index {i}. Skipping.")

            return results
        except Exception as e:
            st.error(f"Error during TF-IDF retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

# --- Loading Function ---
@st.cache_resource # Cache the loaded retriever object
def load_tfidf_retriever(index_path=TFIDF_INDEX_FILE):
    """Loads the pickled TF-IDF retriever object."""
    if not os.path.exists(index_path):
        st.error(f"TF-IDF index file not found at: {index_path}")
        return None
    print(f"Loading TF-IDF retriever from: {index_path}")
    try:
        with open(index_path, 'rb') as f_in:
            # Load the dictionary or object saved during training
            saved_data = pickle.load(f_in)

            # --- Adapt based on how you saved the .pkl file ---
            # Option 1: Saved the entire TfidfRetriever CLASS instance
            if isinstance(saved_data, TfidfRetriever):
                 retriever_obj = saved_data
            # Option 2: Saved a dictionary containing the necessary components
            elif isinstance(saved_data, dict):
                 retriever_obj = TfidfRetriever(
                     vectorizer=saved_data.get('vectorizer'),
                     tfidf_matrix=saved_data.get('tfidf_matrix'),
                     chunk_ids=saved_data.get('chunk_ids'),
                     chunks_dict=saved_data.get('chunks_dict'),
                     original_doc_ids_map=saved_data.get('original_doc_ids_map')
                 )
            else:
                 st.error(f"Loaded TF-IDF pkl file has unexpected format: {type(saved_data)}")
                 return None

            # --- Validation ---
            if not all([retriever_obj.vectorizer, retriever_obj.tfidf_matrix is not None,
                        retriever_obj.chunk_ids, retriever_obj.chunks_dict, retriever_obj.original_doc_ids_map]):
                st.error("Loaded TF-IDF retriever object is missing required components.")
                return None

            print("TF-IDF retriever loaded successfully.")
            return retriever_obj

    except (pickle.UnpicklingError, EOFError) as pe:
        st.error(f"Error unpickling TF-IDF index file (it might be corrupted or incompatible): {pe}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading TF-IDF retriever: {e}")
        import traceback
        traceback.print_exc()
        return None