from huggingface_hub import snapshot_download

# Modeli indirmek için bir dizin belirleyin
model_path = snapshot_download(repo_id="meta-llama/Llama-2-8b-hf", local_dir="models/llama3.1-8B")
print(f"Model indirildi: {model_path}")
