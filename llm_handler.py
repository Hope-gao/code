import streamlit as st
from openai import OpenAI
import os

# --- LLM Configuration ---
MAX_CONTEXT_LENGTH = 5000 # Increased context length slightly
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_TOKENS = 300 # Increased max tokens for potentially longer answers
LLM_TEMPERATURE = 0.7 # Slightly lower temperature for more focused answers

# --- SiliconFlow API Configuration ---
# Use Streamlit Secrets for API Key is strongly recommended
# SILICONFLOW_API_KEY = st.secrets.get("SILICONFLOW_API_KEY", "YOUR_DEFAULT_KEY_HERE")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-xvrbrpercvlfxkfsxsfaidnwpvfjdwouqrxsauhxbdjnkmhh") # Get from env or default (change default)
SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# --- LLM Client Initialization (Cached) ---
@st.cache_resource
def get_llm_client():
    """Initializes and returns the OpenAI client for SiliconFlow."""
    if not SILICONFLOW_API_KEY or SILICONFLOW_API_KEY == "YOUR_DEFAULT_KEY_HERE":
        st.error("SiliconFlow API Key not configured. Please set SILICONFLOW_API_KEY secret or environment variable.")
        return None
    try:
        client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        # Quick check to verify connection and key
        client.models.list()
        print("SiliconFlow OpenAI client initialized and verified successfully.")
        return client
    except Exception as e:
        st.error(f"Error initializing or verifying SiliconFlow OpenAI client: {e}")
        if "Incorrect API key" in str(e):
             st.error("The provided SiliconFlow API Key seems incorrect or invalid.")
        return None

# --- Prompt Building Function (English) ---
def build_prompt(question: str, context_passages_text: list[str]) -> str:
    """
    Builds the prompt for the LLM in English, using provided context.
    """
    if not context_passages_text:
        # Option 1: Tell the LLM there's no context
        context_str = "No relevant context was found."
        # Option 2: Provide minimal context (might hallucinate less)
        # context_str = "Please answer based on general knowledge if no context is provided."
        st.warning("No context retrieved, the LLM might hallucinate or refuse to answer.")
    else:
        # Format context passages clearly
        context_items = []
        current_length = 0
        for i, text in enumerate(context_passages_text):
            item = f"Context Passage {i+1}:\n{text}\n"
            item_len = len(item) # Estimate length
            if current_length + item_len <= MAX_CONTEXT_LENGTH:
                context_items.append(item)
                current_length += item_len
            else:
                # Truncate the last fitting item if needed
                remaining_space = MAX_CONTEXT_LENGTH - current_length
                if remaining_space > 100: # Only add partial if significant space remains
                     context_items.append(item[:remaining_space] + "...\n")
                st.info(f"Context truncated at passage {i+1} due to length limit ({MAX_CONTEXT_LENGTH} chars).")
                break # Stop adding passages
        context_str = "\n".join(context_items)

    # Carefully crafted English prompt
    prompt = f"""You are a helpful Question Answering assistant.
Your task is to answer the given Question based *only* on the provided Context Passages.
- Read the Context Passages carefully.
- Provide a concise and direct answer to the Question.
- If the answer cannot be found within the provided Context Passages, you MUST state: "Based on the provided context, I cannot answer the question."
- Do not use any information outside of the provided context.
- Do not add introductory phrases like "Based on the context..." to your answer unless you cannot find the answer.

Context Passages:
--- START CONTEXT ---
{context_str}
--- END CONTEXT ---

Question: {question}

Answer:"""
    # print("Built Prompt:\n", prompt) # Uncomment for debugging
    return prompt

# --- Answer Generation Function ---
def generate_answer_with_qwen(prompt: str, llm_client) -> str:
    """Generates an answer using the Qwen LLM via SiliconFlow API."""
    if llm_client is None:
        st.error("LLM client is not initialized.")
        return "Error: LLM client not available."

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stream=False # Keep False for simpler handling in Streamlit
        )
        generated_answer = response.choices[0].message.content.strip()

        # Basic cleaning (optional)
        # if generated_answer.startswith('"') and generated_answer.endswith('"'):
        #     generated_answer = generated_answer[1:-1]

        # More robust check for refusals or empty answers
        refusal_phrases = [
            "based on the provided context, i cannot answer",
            "information not found in context",
            "context does not contain",
            "i cannot answer the question",
            "insufficient information",
            # Add Chinese refusals just in case model ignores instructions
            "信息不足", "无法回答", "根据提供的上下文",
        ]
        # Normalize answer for comparison
        normalized_answer = ' '.join(generated_answer.lower().split())
        is_refusal = any(phrase in normalized_answer for phrase in refusal_phrases)

        if not generated_answer or is_refusal or len(generated_answer) < 5 : # Check for empty, refusal, or very short answers
             # Check if context was actually missing, tailor message
             if "No relevant context was found." in prompt:
                 return "No relevant context was found to answer the question."
             else:
                 return "Based on the provided context, I cannot answer the question." # Standard refusal

        return generated_answer

    except Exception as e:
        st.error(f"Error calling SiliconFlow API: {type(e).__name__} - {e}")
        import traceback
        tb_str = traceback.format_exc()
        st.error(f"Traceback:\n{tb_str}")
        if "Incorrect API key" in str(e):
            return "LLM Call Error: Incorrect API key provided."
        elif "limit" in str(e).lower():
            return "LLM Call Error: API rate limit reached."
        else:
            return f"LLM Call Error: An unexpected error occurred."