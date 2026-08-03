"""
Run this ONCE (in Kaggle/Colab, right after training, or locally if you have
the saved_model folder) to publish your fine-tuned model to the Hugging Face Hub.

This is what lets your Streamlit app load the model with just its repo name,
instead of shipping a 400MB+ file inside your GitHub repo.

Steps:
1. pip install huggingface_hub transformers
2. huggingface-cli login   (or set the HF_TOKEN env var / notebook_login())
3. Edit REPO_ID below to "your-username/bert-imdb-sentiment"
4. python push_to_hub.py
5. Copy REPO_ID into MODEL_REPO at the top of app.py
"""

from transformers import BertTokenizer, BertForSequenceClassification

LOCAL_MODEL_DIR = "./saved_model"          # folder produced by model.save_pretrained(...)
REPO_ID = "your-username/bert-imdb-sentiment"  # <-- change this

tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)

tokenizer.push_to_hub(REPO_ID)
model.push_to_hub(REPO_ID)

print(f"Done. Your model now lives at https://huggingface.co/{REPO_ID}")
