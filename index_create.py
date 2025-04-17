# --- 在你的索引创建脚本中 ---
import joblib
import os

# 假设你已经有了处理好的 chunk
# all_chunks = [ (doc_id_1, chunk_text_1, chunk_id_1), (doc_id_1, chunk_text_2, chunk_id_2), ... ]

# --- 构建元数据列表 ---
chunk_metadata = []
for doc_id, chunk_text, chunk_id in all_chunks: # 或者你的循环方式
    chunk_info = {
        'doc_id': doc_id,         # 确保键名是 'doc_id'
        'chunk_text': chunk_text, # 确保键名是 'chunk_text'
        'chunk_id': chunk_id      # 确保键名是 'chunk_id'
    }
    chunk_metadata.append(chunk_info)

# --- 确认保存的是这个列表 ---
INDEX_SAVE_DIR = 'retriever_indices' # 确保路径一致
CHUNK_METADATA_PATH = os.path.join(INDEX_SAVE_DIR, "chunk_metadata.joblib")

if not os.path.exists(INDEX_SAVE_DIR):
    os.makedirs(INDEX_SAVE_DIR)

print(f"Saving chunk metadata to {CHUNK_METADATA_PATH}...")
print(f"Metadata type: {type(chunk_metadata)}") # 应该输出 <class 'list'>
if chunk_metadata:
    print(f"First metadata entry type: {type(chunk_metadata[0])}") # 应该输出 <class 'dict'>
    print(f"First metadata entry keys: {chunk_metadata[0].keys()}") # 应该包含 'doc_id', 'chunk_text', 'chunk_id'

# --- 执行保存 ---
joblib.dump(chunk_metadata, CHUNK_METADATA_PATH)
print("Chunk metadata saved.")

# --- 同时也要保存 FAISS index ---
# faiss.write_index(faiss_index, FASTTEXT_FAISS_INDEX_PATH)