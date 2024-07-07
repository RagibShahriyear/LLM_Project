from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload

# Path to your service account credentials JSON file
cred_path = 'creds/pydrive-428612-ce86bb7e142a.json'

# Authenticate and create the service
creds = Credentials.from_service_account_file(cred_path, scopes=['https://www.googleapis.com/auth/drive'])
service = build('drive', 'v3', credentials=creds)

def download_file_from_drive(file_id, destination):
    """
    Download a file from Google Drive.

    Args:
        file_id (str): The ID of the file to download.
        destination (str): The path to save the downloaded file.
    """
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    print(f"File downloaded to {destination}")

# Example usage
large_embedding_file_id = '1VA3yFn_KzEHHeW___5Y8LLkVVfhllEc-'
small_embedding_file_id = '1lfi94Zxp5JOQJg-SuJijgtlWwrMk8Nwo'

file_id = large_embedding_file_id
destination = 'downloaded_embeddings.npy'
download_file_from_drive(file_id, destination)

# ---------------------------------------------------------------

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Load embeddings and corresponding documents

# Load the guidelines dataset
# Specify the dataset name and the column containing the content
from datasets import load_dataset
ds = load_dataset("jpcorb20/rag_epfl_guidelines")
guidelines = ds['train']['text']


embeddings = np.load('downloaded_embeddings.npy')
# with open('path/to/documents.pkl', 'rb') as f:


# Load pre-trained model and tokenizer for query embedding
model_name = "NeuML/pubmedbert-base-embeddings-matryoshka"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Set device (use GPU if available)
device = torch.device("mps")
model.to(device)

def get_relevant_documents(query, embeddings, documents, top_k=5):
    # Compute the embedding for the query
    query_embedding = embed_text(query, tokenizer, model, device)
    
    # Compute cosine similarity between query embedding and document embeddings
    similarities = cosine_similarity(query_embedding, embeddings)
    
    # Get the top K most similar documents
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]
    relevant_documents = [documents[i] for i in top_k_indices]
    
    return relevant_documents

# Example usage
query = "lung nodules management"
relevant_docs = get_relevant_documents(query, embeddings, guidelines)
for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:\n{doc}\n")
    



