# app.py
import streamlit as st
import torch
import os
import time

# --- Import modular components ---
from dpr_retriever import (
    load_dpr_question_model, load_faiss_index, load_passage_mapping,
    DprPassageFaissRetriever, QUESTION_ENCODER_NAME,
    PASSAGE_FAISS_INDEX_PATH, PASSAGE_MAPPING_PATH
)
from tfidf_retriever import load_tfidf_retriever, TfidfRetriever  # Import class
from bm25_retriever import load_bm25_retriever, Bm25Retriever     # Import class
from fasttext_retriever import load_fasttext_retriever, FastTextFaissRetriever # Import class
from llm_handler import get_llm_client, build_prompt, generate_answer_with_qwen

# --- NLTK Setup (Run ONCE Globally) ---
import nltk
import setup_nltk # Assuming setup_nltk.py handles downloads robustly

try:
    # Check if core components are available after setup attempt
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print("NLTK components (punkt, stopwords) found.")
    NLTK_READY = True
except LookupError as e:
    st.error(f"NLTK setup incomplete: {e}. Some retrievers might fail. Please ensure NLTK data is downloaded correctly (check setup_nltk.py or run manually).")
    NLTK_READY = False
# ------------------------------------


# --- General Configuration ---
NUM_CONTEXT_PASSAGES = 5  # How many passages to display in UI expanders (can be same or different)

# --- Device Selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (Note: FastText/BM25/TFIDF might primarily use CPU)")

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Intelligent Q&A System (COMP5423)")
st.markdown("Select a retrieval model and ask a question.")

# --- Sidebar for Model Selection ---
st.sidebar.header("Configuration")
retriever_choice = st.sidebar.selectbox(
    "Choose Retriever Model:",
    ("DPR", "BM25", "TF-IDF", "FastText")
)

# --- Load Selected Retriever (Cached) ---
@st.cache_resource(show_spinner="Loading selected retriever...")
def load_selected_retriever(choice):
    if not NLTK_READY and choice in ["TF-IDF", "BM25", "FastText"]:
         st.error(f"Cannot load {choice} because NLTK setup failed. Please fix NLTK issues.")
         return None

    if choice == "DPR":
        q_tokenizer, q_encoder = load_dpr_question_model(QUESTION_ENCODER_NAME, device)
        faiss_index = load_faiss_index()
        passage_mapping = load_passage_mapping()
        if not all([q_tokenizer, q_encoder, faiss_index, passage_mapping]):
            st.error("Failed to load one or more DPR components.")
            return None
        return DprPassageFaissRetriever(q_tokenizer, q_encoder, faiss_index, passage_mapping, device)

    elif choice == "TF-IDF":
        retriever = load_tfidf_retriever()
        if retriever is None:
             st.error("Failed to load TF-IDF retriever.")
        return retriever

    elif choice == "BM25":
        retriever = load_bm25_retriever()
        if retriever is None:
            st.error("Failed to load BM25 retriever.")
        return retriever

    elif choice == "FastText":
        retriever = load_fasttext_retriever()
        if retriever is None:
             st.error("Failed to load FastText retriever.")
        return retriever

    else:
        st.error("Invalid retriever choice selected.")
        return None

selected_retriever = load_selected_retriever(retriever_choice)

# --- Load LLM Client ---
llm_client = get_llm_client() # From llm_handler.py

# --- Main Interaction Area ---
if selected_retriever and llm_client:
    st.success(f"{retriever_choice} Retriever and LLM ready!")

    st.header("Ask your Question:")
    user_question = st.text_input("Question:", key="question_input")

    if st.button("Submit", key="submit_button"):
        if user_question:
            st.markdown("---")
            st.header("Results")

            # --- 1. Retrieval ---
            start_time = time.time()
            with st.spinner(f"Retrieving passages using {retriever_choice}..."):
                # Calls the standardized retrieve_passages method
                retrieved_passages = selected_retriever.retrieve_passages(
                    user_question,
                    top_k=NUM_CONTEXT_PASSAGES # Retrieve enough for context
                )
            end_time = time.time()
            st.caption(f"Retrieval took {end_time - start_time:.2f} seconds.")

            # --- Display Retrieved Passages ---
            # Display slightly fewer if desired, e.g., top 5, even if 7 were retrieved for LLM
            num_to_display = min(len(retrieved_passages), 5) # Example: Display top 5
            st.subheader(f"Top {num_to_display} Context Passages ({retriever_choice}):")

            if retrieved_passages:
                context_texts = [] # Collect texts for LLM from ALL retrieved
                for p in retrieved_passages: # Iterate through all retrieved for context
                     context_texts.append(p['passage_text'])

                # Display only the top N passages in the UI
                for i, p in enumerate(retrieved_passages[:num_to_display]):
                    # Determine ID to display (prefer chunk_id if available, else FAISS index)
                    display_id = p.get('chunk_id', f"faiss_idx_{p.get('passage_index_in_faiss', 'N/A')}")
                    with st.expander(f"Passage {i+1} (Score: {p['score']:.4f}, Source Doc ID: {p.get('doc_id', 'N/A')}, ID: {display_id})"):
                        st.write(p['passage_text'])
            else:
                st.warning(f"{retriever_choice} did not retrieve any relevant passages.")
                context_texts = []

            # --- 2. Answer Generation ---
            st.subheader("Generated Answer:")
            if not context_texts:
                 st.warning("Skipping answer generation as no context was retrieved.")
                 st.markdown("**Answer:** Based on the provided context, I cannot answer the question.")
            else:
                with st.spinner("Generating answer using Qwen LLM..."):
                    # a. Build Prompt (using English version from llm_handler)
                    start_time = time.time()
                    # Use the potentially larger list of context_texts (up to NUM_CONTEXT_PASSAGES)
                    prompt = build_prompt(user_question, context_texts)
                    with st.expander("Show Prompt sent to LLM"):
                       st.text(prompt)

                    # b. Call LLM
                    generated_answer = generate_answer_with_qwen(prompt, llm_client)
                    end_time = time.time()
                    st.caption(f"LLM generation took {end_time - start_time:.2f} seconds.")

                    # c. Display Answer
                    st.markdown(f"**Answer:**")
                    st.markdown(generated_answer)

        else:
            st.warning("Please enter a question.")

elif not selected_retriever:
     st.error(f"Failed to load the selected retriever ({retriever_choice}). Please check logs, file paths, and NLTK setup.")
elif not llm_client:
     st.error("Failed to initialize the LLM client. Please check API Key and configuration.")


# --- Footer ---
st.markdown("---")
st.caption(f"Using {retriever_choice} for retrieval and Qwen LLM for answer generation.")