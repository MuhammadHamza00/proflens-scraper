from flask import Flask, json, request, render_template, jsonify
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
import os
app = Flask(__name__)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Create or connect to the Pinecone index
index = pc.Index("rmp")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        url = request.form['url']
        if url:
            data = scrape_professor_data(url)
            
            if data:
                process_and_store_data(data)
                message = "Data scraped and stored successfully."
            else:
                message = "Failed to scrape the data."
        else:
            message = "No URL provided."
        return render_template('result.html', message=message)
    return render_template('index.html')

def scrape_professor_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    
    # Initialize fields with defaults
    professor_name = "Richard Page"  
    review_text = "No reviews available."
    subject = "Orthopaedic Surgery"
    institution = "Geelong Orthopaedics Group"
    stars = "Not available"  # This is usually a feature of review sites, so it's not applicable here

    # Extracting the professor's name
    name_tag = soup.find('h1')
    if name_tag:
        professor_name = name_tag.text.strip()


    review_text = ""
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        review_text += p.text.strip() + " "

    return {
        "professor": professor_name,
        "review": review_text.strip(),
        "subject": subject,
        "institution": institution,
        "stars": stars,
    }


def process_and_store_data(data):
    # Concatenate the data fields into a single string
    combined_text = f"{data['professor']} {data['review']} {data['subject']} {data['institution']}"
    
    # Generate the embeddings for the combined text
    embedding_768 = model.encode(combined_text)
    
    # Concatenate the embedding to get a 1536-dimensional embedding
    embedding_1536 = list(embedding_768) + list(embedding_768)
    embedding_1536 = [float(val) for val in embedding_1536]

    # Use professor's name as vector ID
    vector_id = data['professor']

    # Prepare the data for upsert
    upsert_response = index.upsert(
        vectors=[
            {
                "id": vector_id, 
                "values": embedding_1536, 
                "metadata": {
                    "institution": data["institution"],
                    "review": data["review"],
                    "stars": data["stars"],
                    "subject": data["subject"]
                }
            }
        ],
        namespace="ns1"
    )
    
    print(f"Upserted count: {upsert_response['upserted_count']}")
    # Print index statistics
    print(index.describe_index_stats())

if __name__ == '__main__':
    app.run(debug=True)
