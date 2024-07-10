import time
from transformers import pipeline, AutoTokenizer
import torch

# Load zero-shot classification pipeline with BART
device = 0 if torch.cuda.is_available() else -1
start_time = time.time()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cuda")
classifier_load_time = time.time() - start_time

# Load the tokenizer
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
tokenizer_load_time = time.time() - start_time

# Read job description
with open('JD.txt', 'r') as file:
    job_description = file.read()

# Define possible categories
categories = ["Programming Languages", "Cloud Technologies"]


# Function to tokenize the text
def tokenize_text(text):
    start_time = time.time()
    tokens = tokenizer.tokenize(text)
    clean_tokens = [token.replace('Ġ', '').replace('Ċ', '') for token in tokens if token.strip()]
    clean_tokens = [token for token in clean_tokens if token]
    tokenize_time = time.time() - start_time
    print(f"Tokenization time: {tokenize_time:.2f} seconds")
    return clean_tokens


# Function to classify each token
def classify_tokens(tokens, categories):
    results = {category: [] for category in categories}
    start_time = time.time()
    for token in tokens:
        classification = classifier(token, candidate_labels=categories)
        best_label = classification['labels'][0]
        if classification['scores'][0] >= 0.9:
            results[best_label].append(token)
    classify_time = time.time() - start_time
    print(f"Classification time: {classify_time:.2f} seconds")
    return results


# Tokenize the job description
tokens = tokenize_text(job_description)

# Classify each token
classified_tokens = classify_tokens(tokens, categories)

# Display the results
for category, tokens in classified_tokens.items():
    print(f"{category}: {tokens}")

# Print load times
print(f"Classifier load time: {classifier_load_time:.2f} seconds")
print(f"Tokenizer load time: {tokenizer_load_time:.2f} seconds")
