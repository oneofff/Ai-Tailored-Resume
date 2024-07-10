import time
from transformers import pipeline, AutoTokenizer
import torch

# Load zero-shot classification pipeline with BART
device = 0 if torch.cuda.is_available() else -1
start_time = time.time()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
classifier_load_time = time.time() - start_time

# Load the tokenizer
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
tokenizer_load_time = time.time() - start_time

# Define job description
job_description = ('''CANNOT DO C2C

 Description 

Looking for a seasoned Java/J2EE Developer Lead with 10+ years progressive industry experience

 Bachelor’s degree or equivalent in Engineering, Computer Science, Management Information Systems, Computer Information Systems, or related
 Analyzing, designing, and developing enterprise applications with Agile development methodology
 Designing, building, and maintaining Mule Soft and J2EE based Integration and API platforms
 Identify, analyze and develop Integration flows using Mule Connectors, Design Center and API Management
 Assisting in web services development with associated skills using SOAP, REST and HTTP in a high stakes/high volume environment
 Experience with Marklogic or other NoSQL Databases
 Working knowledge of GIT Hub or similar environment
 Working closely with product partners to on board the new functional requirements in the system;
 Assisting with the integration into Mule Soft platform when necessary, setup CI/ CD routines, and configure Mule Soft runtimes;
 Participating in code reviews and daily SCRUM;


 Skills 

Java, rest api, j2ee, json, springboot, microservices, rest, api, javascript, spring

 Top Skills Details 

Java,rest api,j2ee,json

 Additional Skills & Qualifications 

Develop and support Java based application for CashPro Admin Projects

 10+ years of development experience in Java/J2EE
 5+ years of work experience in an Agile environment. Work history of participating in daily agile routines and estimation of stories with minimal direction


Expertise in complex SQL and query plans AWS

 Experience Level 

Intermediate Level

 About TEKsystems 

We're partners in transformation. We help clients activate ideas and solutions to take advantage of a new world of opportunity. We are a team of 80,000 strong, working with over 6,000 clients, including 80% of the Fortune 500, across North America, Europe and Asia. As an industry leader in Full-Stack Technology Services, Talent Services, and real-world application, we work with progressive leaders to drive change. That's the power of true partnership. TEKsystems is an Allegis Group company.

The company is an equal opportunity employer and will consider all applications without regards to race, sex, age, color, religion, national origin, veteran status, disability, sexual orientation, gender identity, genetic information or any characteristic protected by law.''')

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
