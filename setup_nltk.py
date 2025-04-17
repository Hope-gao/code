import nltk

# 列出所有你的应用需要的 NLTK 数据包
required_nltk_packages = [
    'punkt',        # 用于 word_tokenize
    'stopwords',    # 英文停用词
    'wordnet',      # 用于 WordNetLemmatizer (如果你选择使用)
    'omw-1.4',       # WordNet 的开放多语言部分 (如果你选择使用 WordNet)
    'punkt_tab'   # 用于分词的 punkt 数据包 (如果你选择使用)
]

print(f"开始下载 NLTK 数据包: {required_nltk_packages}")
print(f"NLTK 将尝试下载到默认路径: {nltk.data.path}")

download_count = 0
error_count = 0

for package_id in required_nltk_packages:
    try:
        print(f"\n正在检查/下载 '{package_id}'...")
        # 使用 raise_on_error=True 可以在下载失败时抛出异常
        nltk.download(package_id, quiet=False, raise_on_error=True)
        print(f"'{package_id}' 下载/验证成功。")
        download_count += 1
    except ValueError as ve:
         # Handle case where package might already be up-to-date and download returns None/False but isn't an Exception
         # Check if it can be found now
         try:
             resource_path = f'tokenizers/{package_id}' if package_id=='punkt' else f'corpora/{package_id}'
             if package_id in ['wordnet', 'omw-1.4']: resource_path = f'corpora/{package_id}'
             nltk.data.find(resource_path)
             print(f"'{package_id}' 已存在或刚刚下载成功。")
             download_count +=1 # Count as success if found after download attempt
         except LookupError:
             print(f"--- 下载/查找 '{package_id}' 时出错 (ValueError): {ve}. 请检查包名是否正确。")
             error_count += 1
    except Exception as e:
        print(f"--- 下载 '{package_id}' 时发生错误 ---")
        print(f"错误信息: {e}")
        print(f"请检查网络连接和NLTK数据目录的写入权限。")
        print(f"默认搜索路径: {nltk.data.path}")
        error_count += 1

print("\n-------------------------------------")
if error_count == 0:
    print(f"所有 {download_count} 个必需的 NLTK 数据包均已成功下载或已存在。")
else:
    print(f"*** {error_count} 个 NLTK 数据包下载失败。请查看上面的错误信息。***")
print("-------------------------------------")