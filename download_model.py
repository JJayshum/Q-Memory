from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-3B-Instruct", cache_dir="/root/autodl-tmp/modelscope_cache"))
