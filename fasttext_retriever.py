# fasttext_retriever.py
import streamlit as st
import pickle
import os
import re
import time
import numpy as np
import setup_nltk # Ensure NLTK setup runs if needed globally
import faiss
import joblib # For loading .joblib file
from gensim.models import KeyedVectors # For loading FastText model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()
    nltk_ready = True
except LookupError:
    st.error("NLTK 'stopwords' or 'punkt' data missing. Please ensure NLTK setup runs in app.py.")
    nltk_ready = False

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(__file__)
# Assuming indices are in 'retriever_indices' directory relative to this script
INDEX_SAVE_DIR = os.path.join(SCRIPT_DIR, 'retriever_indices')

# --- Model File Paths ---
FASTTEXT_MODEL_PATH = os.path.join(INDEX_SAVE_DIR, "wiki-news-300d-1M.vec")

# --- Index File Paths ---
FASTTEXT_FAISS_INDEX_PATH = os.path.join(INDEX_SAVE_DIR, "fasttext_index.faiss")
CHUNK_METADATA_PATH = os.path.join(INDEX_SAVE_DIR, "chunk_metadata.joblib")

# --- Print paths for debugging ---
print(f"[fasttext_retriever.py] FastText Model Path: {FASTTEXT_MODEL_PATH}")
print(f"[fasttext_retriever.py] FAISS Index Path: {FASTTEXT_FAISS_INDEX_PATH}")
print(f"[fasttext_retriever.py] Chunk Metadata Path: {CHUNK_METADATA_PATH}")

# --- Preprocessing Functions ---
def preprocess_text_fasttext(text):
    """Preprocesses text for FastText: remove HTML, non-alpha, lowercase, remove stopwords, return tokens."""
    if not isinstance(text, str) or not nltk_ready: return []
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

# --- Average Vector Calculation ---
def get_average_vector(tokens, model, vector_size):
    """Calculates the average FastText vector for a list of tokens."""
    vectors = []
    for token in tokens:
        try:
            # Use model.get_vector(token) which handles OOV better for some Gensim versions
            # or direct access model[token] if get_vector isn't available/needed
            vectors.append(model[token])
            # vectors.append(model.get_vector(token)) # Alternative
        except KeyError:
            # Ignore words not in vocabulary
            pass
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# --- Loading Functions (Cached) ---
@st.cache_resource(show_spinner="Loading FastText model...")
def load_fasttext_model(model_path):
    """Loads the pre-trained FastText model."""
    if not os.path.exists(model_path):
        st.error(f"FastText model file not found: {model_path}")
        return None
    print(f"Loading FastText model from: {model_path}")
    try:
        # Determine format based on extension (heuristic)
        is_binary = model_path.endswith('.bin')
        model = KeyedVectors.load_word2vec_format(model_path, binary=is_binary)
        print(f"FastText model loaded. Vocab size: {len(model.key_to_index)}, Vector size: {model.vector_size}")
        return model
    except Exception as e:
        st.error(f"Error loading FastText model: {e}")
        return None

@st.cache_resource(show_spinner="Loading FastText FAISS index...")
def load_faiss_index_ft(index_path=FASTTEXT_FAISS_INDEX_PATH):
    """Loads FAISS index for FastText."""
    if not os.path.exists(index_path):
        st.error(f"FastText FAISS index file not found: {index_path}")
        return None
    print(f"Loading FastText FAISS index: {index_path}")
    try:
        index = faiss.read_index(index_path)
        print(f"FastText FAISS index loaded successfully. Index size: {index.ntotal}")
        return index
    except Exception as e:
        st.error(f"Error loading FastText FAISS index: {e}")
        return None

# Use cache_data for potentially large metadata object
@st.cache_data(show_spinner="Loading chunk metadata...")
def load_chunk_metadata(metadata_path=CHUNK_METADATA_PATH):
    """Loads chunk metadata from joblib file."""
    if not os.path.exists(metadata_path):
        st.error(f"Chunk metadata file not found: {metadata_path}")
        return None
    print(f"Loading chunk metadata: {metadata_path}")
    try:
        metadata = joblib.load(metadata_path)
        print(f"Chunk metadata loaded successfully. Type: {type(metadata)}, Example entry (if list/dict): {metadata[0] if isinstance(metadata, list) and metadata else 'N/A'}")
        # Add more validation based on expected structure (e.g., list of dicts)
        if isinstance(metadata, list) and metadata:
            if not isinstance(metadata[0], dict) or not all(k in metadata[0] for k in ['doc_id', 'chunk_text', 'chunk_id']):
                 st.warning("Metadata format validation failed: Expected list of dicts with 'doc_id', 'chunk_text', 'chunk_id'.")
        elif not isinstance(metadata, list): # Adjust if it's a dict keyed by index
             st.warning(f"Metadata format validation failed: Expected list, got {type(metadata)}.")

        return metadata
    except Exception as e:
        st.error(f"Error loading chunk metadata: {e}")
        return None

# --- FastText + FAISS Retriever Class ---
class FastTextFaissRetriever:
    def __init__(self, ft_model, faiss_index, chunk_metadata):
        self.ft_model = ft_model
        self.faiss_index = faiss_index
        self.chunk_metadata = chunk_metadata  # Assumed to be a list of dicts indexed 0..N-1
        self.vector_size = ft_model.vector_size
        self.using_chunks = True  # Indicates this retriever works with chunks

        # Validation
        if self.faiss_index and self.chunk_metadata and self.faiss_index.ntotal != len(self.chunk_metadata):
            st.warning(f"FAISS index size ({self.faiss_index.ntotal}) != chunk metadata size ({len(self.chunk_metadata)})!")

    # Standardized retrieval method
    def retrieve_passages(self, query, top_k=5):
        """Retrieves top-k passages using FastText average vector and FAISS."""
        if not all([query, self.ft_model, self.faiss_index, self.chunk_metadata]):
            st.error("FastText retriever components incomplete.")
            return []
        if not nltk_ready:
            st.error("NLTK not ready, cannot preprocess query for FastText.")
            return []

        try:
            # 1. Preprocess query
            tokenized_query = preprocess_text_fasttext(query)
            if not tokenized_query:
                st.warning("Preprocessing removed the entire query for FastText.")
                return []

            # 2. Calculate query vector
            query_vector = get_average_vector(tokenized_query, self.ft_model, self.vector_size)
            if np.all(query_vector == 0):
                st.warning("Query vector is zero (all words might be OOV).")
                return []

            # 3. Normalize query vector (IMPORTANT: Assumes FAISS index uses normalized vectors - e.g., IndexFlatIP)
            norm = np.linalg.norm(query_vector)
            if norm == 0:  # Should not happen if check above passed, but safety first
                 st.warning("Query vector norm is zero after calculation.")
                 return []
            normalized_query_vector = (query_vector / norm).astype('float32').reshape(1, -1) # Reshape for FAISS

            # 4. FAISS Search
            # Scores are likely inner products (higher is better) if normalized + IndexFlatIP
            scores, indices = self.faiss_index.search(normalized_query_vector, top_k)

            # 5. Map indices to metadata
            results = []
            faiss_indices = indices[0]
            faiss_scores = scores[0]

            for i, idx in enumerate(faiss_indices):
                idx = int(idx) # Ensure integer index
                if 0 <= idx < len(self.chunk_metadata):
                    try:
                        metadata_entry = self.chunk_metadata[idx]
                        # Assuming metadata is a list of dicts {'doc_id': ..., 'chunk_text': ..., 'chunk_id': ...}
                        if isinstance(metadata_entry, dict):
                            results.append({
                                "doc_id": metadata_entry.get("doc_id", -1), # Provide default
                                "passage_text": metadata_entry.get("chunk_text", "Error: Chunk text not found"),
                                "score": float(faiss_scores[i]), # Inner product score
                                "passage_index_in_faiss": idx, # Index in FAISS/metadata list
                                "chunk_id": metadata_entry.get("chunk_id", f"chunk_{idx}") # Get or create chunk ID
                            })
                        else:
                             st.warning(f"Metadata entry at index {idx} is not a dictionary: {metadata_entry}")
                    except Exception as map_e:
                        st.warning(f"Error processing metadata for index {idx}: {map_e}")
                else:
                    st.warning(f"FAISS returned out-of-bounds index {idx} for metadata size {len(self.chunk_metadata)}")

            return results

        except Exception as e:
            st.error(f"Error during FastText retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

# --- Top-Level Loading Function ---
@st.cache_resource(show_spinner="Initializing FastText Retriever...")
def load_fasttext_retriever():
    """Loads all components and initializes the FastTextFaissRetriever."""
    print("Attempting to load FastText retriever components...")
    ft_model = load_fasttext_model(FASTTEXT_MODEL_PATH)
    faiss_index = load_faiss_index_ft() # Uses default path
    chunk_metadata = load_chunk_metadata() # Uses default path

    if not all([ft_model, faiss_index, chunk_metadata]):
        st.error("Failed to load one or more FastText retriever components. Cannot initialize retriever.")
        return None

    # Perform validation after loading all components
    if faiss_index.ntotal != len(chunk_metadata):
         st.error(f"Critical Error: FAISS index size ({faiss_index.ntotal}) does not match metadata size ({len(chunk_metadata)}). Cannot proceed.")
         return None
    if faiss_index.d != ft_model.vector_size:
         st.error(f"Critical Error: FAISS index dimension ({faiss_index.d}) does not match FastText model dimension ({ft_model.vector_size}). Cannot proceed.")
         return None


    print("All FastText components loaded, initializing retriever class.")
    retriever = FastTextFaissRetriever(ft_model, faiss_index, chunk_metadata)
    print("FastTextFaissRetriever initialized successfully.")
    return retriever

# --- END OF FILE fasttext_retriever.py ---