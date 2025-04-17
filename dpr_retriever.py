# dpr_retriever.py
import streamlit as st
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import faiss
import numpy as np
import time
import os
import json

# --- Configuration (relative paths or configurable) ---
SCRIPT_DIR = os.path.dirname(__file__)
INDEX_DIR = os.path.join(SCRIPT_DIR, 'retriever_indices')

QUESTION_ENCODER_NAME = 'facebook/dpr-question_encoder-single-nq-base'
# --- Build DPR file path---
PASSAGE_FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "dpr_passage_faiss_index.idx")
PASSAGE_MAPPING_PATH = os.path.join(INDEX_DIR, "dpr_passage_mapping.json")
print(f"[dpr_retriever.py] Calculated FAISS Index Path: {PASSAGE_FAISS_INDEX_PATH}")
print(f"[dpr_retriever.py] Calculated Mapping Path: {PASSAGE_MAPPING_PATH}")
MAX_LENGTH = 512  # DPR tokenizer max length

# --- Model Loading Functions (Cached) ---
@st.cache_resource
def load_dpr_question_model(q_encoder_name, device):
    """Loads DPR Question Encoder and Tokenizer."""
    print(f"Loading DPR Question Tokenizer: {q_encoder_name}")
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_encoder_name)
    print(f"Loading DPR Question Encoder: {q_encoder_name}")
    q_encoder = DPRQuestionEncoder.from_pretrained(q_encoder_name)
    q_encoder.to(device).eval() # Move to specified device
    print("DPR Question model loaded.")
    return q_tokenizer, q_encoder

@st.cache_resource
def load_faiss_index(index_path=PASSAGE_FAISS_INDEX_PATH):
    """Loads FAISS index."""
    if not os.path.exists(index_path):
        st.error(f"FAISS index file not found: {index_path}")
        return None
    print(f"Loading FAISS index: {index_path}")
    try:
        index = faiss.read_index(index_path)
        print("FAISS index loaded successfully.")
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

@st.cache_data # Use cache_data for potentially large JSON/list
def load_passage_mapping(mapping_path=PASSAGE_MAPPING_PATH):
    """Loads passage mapping file."""
    if not os.path.exists(mapping_path):
        st.error(f"Passage mapping file not found: {mapping_path}")
        return None
    print(f"Loading passage mapping: {mapping_path}")
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print("Passage mapping loaded successfully.")
        # Basic validation (can be expanded)
        if not isinstance(mapping, list):
             st.warning(f"Expected passage_mapping to be a list, got {type(mapping)}.")
        elif mapping and not (isinstance(mapping[0], list) and len(mapping[0]) == 2):
             st.warning(f"Expected passage_mapping elements like [doc_id, passage_text], got {mapping[0]}.")
        return mapping
    except Exception as e:
        st.error(f"Error loading passage mapping: {e}")
        return None

# --- DPR Retriever Class (Adapted from original app.py) ---
class DprPassageFaissRetriever:
    def __init__(self, q_tokenizer, q_encoder, faiss_index, passage_mapping, device):
        self.q_tokenizer = q_tokenizer
        self.q_encoder = q_encoder
        self.faiss_index = faiss_index
        self.passage_mapping = passage_mapping # This is the loaded list [[doc_id, passage_text], ...]
        self.device = device
        self.vector_dim = q_encoder.config.hidden_size if q_encoder else None
        self.max_length = MAX_LENGTH
        print(f"DPR Passage Retriever initialized. Index size: {self.faiss_index.ntotal if self.faiss_index else 'N/A'}")
        if self.faiss_index and self.passage_mapping and self.faiss_index.ntotal != len(self.passage_mapping):
            st.warning(f"FAISS index size ({self.faiss_index.ntotal}) != passage mapping size ({len(self.passage_mapping)})!")

    # This method already provides the desired interface
    def retrieve_passages(self, query, top_k=5):
        """Retrieves top-k passages with scores using DPR and FAISS."""
        if not all([query, self.faiss_index, self.q_encoder, self.q_tokenizer, self.passage_mapping, self.vector_dim]):
            st.error("DPR retriever components incomplete.")
            return []
        try:
            inputs = self.q_tokenizer(query, max_length=self.max_length, padding=False, truncation=True, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.q_encoder(**inputs)
                query_vector = outputs.pooler_output

            # DPR vectors in FAISS are typically normalized, so normalize query vector too
            norm = torch.linalg.norm(query_vector, dim=1, keepdim=True)  # Calculate norm along the embedding dimension
            normalized_query_vector = query_vector / (norm + 1e-8)
            query_np = normalized_query_vector.cpu().numpy().astype('float32')  # Keep shape (1, dim)


            #  FAISS search
            scores, indices = self.faiss_index.search(query_np, top_k)

            results = []
            valid_indices = indices[0]  # shape: (top_k,)
            valid_scores = scores[0]  # shape: (top_k,)

            for i, idx in enumerate(valid_indices):
                idx = int(idx)  # FAISS indices can sometimes be int64
                if 0 <= idx < len(self.passage_mapping):
                    try:
                        mapping_entry = self.passage_mapping[idx]
                        if isinstance(mapping_entry, list) and len(mapping_entry) == 2:
                            original_doc_id, passage_text = mapping_entry
                            # Ensure doc_id is hashable (int or str usually)
                            try:
                                # Attempt conversion if needed, adjust based on your actual doc_id type
                                doc_id_processed = int(original_doc_id)
                            except (ValueError, TypeError):
                                doc_id_processed = str(original_doc_id) # Fallback to string


                            results.append({
                                "doc_id": doc_id_processed,
                                "passage_text": passage_text,
                                "score": float(valid_scores[i]),
                                "passage_index_in_faiss": idx,
                                "chunk_id": f"dpr_passage_{idx}"  # Create a pseudo-chunk ID if needed
                            })
                        else:
                            st.warning(f"DPR Passage mapping index {idx} format mismatch: {mapping_entry}")
                    except Exception as map_e:
                        st.warning(f"Error processing DPR passage mapping index {idx}: {map_e}")
                else:
                    st.warning(f"FAISS returned out-of-bounds index {idx} for mapping size {len(self.passage_mapping)}")

            return results

        except Exception as e:
            st.error(f"Error during DPR retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []