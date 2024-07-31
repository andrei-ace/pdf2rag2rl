import torch
from transformers import DistilBertTokenizer, DistilBertModel
EMBEDDING_DIM = 768

# Load the text embedding model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
text_model.to(device)

def get_text_embeddings(texts, tokenizer=tokenizer, model=text_model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Get the mean embeddings
    return embeddings