import time
from transformers import pipeline, AutoTokenizer
import torch
from datasets import Dataset

# Load zero-shot classification pipeline with BART
device = 0 if torch.cuda.is_available() else -1
start_time = time.time()
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device="cuda",
                      truncation=True,
                      batch_size=128
                      )
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
def tokenize_text(job_description):
    start_time = time.time()
    tokens = tokenizer.tokenize(job_description)
    clean_tokens = [token.replace('Ġ', '').replace('Ċ', '') for token in tokens if token.strip()]
    clean_tokens = [token for token in clean_tokens if token]
    tokenize_time = time.time() - start_time
    print(f"Tokenization time: {tokenize_time:.2f} seconds")
    return clean_tokens


def create_ds(tokens):
    start_time = time.time()
    dataset = Dataset.from_dict({"tokens": tokens})
    ds_create_time = time.time() - start_time
    print(f"Dataset create time: {ds_create_time:.2f} seconds")
    return dataset


# Function to classify each token
def classify_tokens(ds, categories):
    start_time = time.time()
    classification = classifier(ds, candidate_labels=categories)
    classify_time = time.time() - start_time
    print(classification)
    print(f"Classification time: {classify_time:.2f} seconds")
    return classification


# Tokenize the job description
tokens = tokenize_text(job_description)
ds = create_ds(tokens)
# Classify each token
classified_tokens = classify_tokens(tokens, categories)

results = {category: [] for category in categories}

# Display the results
for token in classified_tokens:
    best_label = token['labels'][0]
    if token['scores'][0] >= 0.9:
        results[best_label].append(token['sequence'])

print(results)

# Print load times
print(f"Classifier load time: {classifier_load_time:.2f} seconds")
print(f"Tokenizer load time: {tokenizer_load_time:.2f} seconds")
