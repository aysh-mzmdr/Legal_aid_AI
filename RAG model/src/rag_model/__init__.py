import os

# Fully offline: the embedding model loads from the local Hugging Face cache and never contacts
# the Hub, and ChromaDB sends no telemetry. Set here so it applies before any submodule imports
# huggingface_hub / transformers. Run once with RAG_ALLOW_DOWNLOAD=1 to download a new model.
if os.environ.get("RAG_ALLOW_DOWNLOAD") != "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # ChromaDB
