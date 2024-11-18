# Import required modules
import pandas as pd
import json
from   transformers import LongformerModel, LongformerTokenizer
import torch
from   tqdm import tqdm
from   datetime import datetime

# Load the tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare dataset
filename = 'reviews_sample_25000.json'
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)
#selected_columns = ['stars', 'text']
#df = pd.DataFrame(data)[selected_columns]
texts = [rec['text'] for rec in data]
labels = [int(rec['stars'])-1 for rec in data] # Convert stars to 0-based integers

def extract_embeddings(texts, tokenizer, model, device):
    embeddings = []
    for text in tqdm(texts):
        # Tokenize the text
        inputs = tokenizer(text, max_length=4096, padding="max_length", truncation=True, return_tensors="pt")

        # Move inputs to the device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the Longformer model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract CLS token embedding (or use other pooling methods)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)

    return embeddings

embeddings = extract_embeddings(texts, tokenizer, model, device)

# Convert embeddings to a DataFrame
X = pd.DataFrame(embeddings)

# Save to Parquet
X.to_parquet(f"longformer_embeddings_{datetime.now().strftime('%Y%m%d%H%M%S')}.parquet", index=False)
