from transformers import DistilBertTokenizer, DistilBertModel
EMBEDDING_DIM = 768

# Load the text embedding model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")


def get_text_embeddings(texts, tokenizer=tokenizer, model=text_model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Get the mean embeddings
    return embeddings