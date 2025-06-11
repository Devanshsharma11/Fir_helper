from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Global variables for model and dataset
model = None
new_ds = None

def load_model_and_data():
    global model, new_ds
    
    # Load preprocessed data
    try:
        new_ds = pickle.load(open('preprocess_data.pkl', 'rb'))
        print("Preprocessed data loaded successfully")
    except FileNotFoundError:
        print("Preprocessed data not found. Loading and processing raw data...")
        # Load and preprocess data
        try:
            ds = pd.read_csv('/Users/devanshsharma/Downloads/FIR-DATA.csv')
            print(ds.head(2))
        except UnicodeDecodeError as err:
            print(f"Error: {err}")
            return False

        ds.fillna("Not mentioned", inplace=True)
        ds['combo'] = ds['Description'] + " " + ds['Offense']
        ds['combo'] = ds['combo'].apply(preprocess_text)

        # Save and Load Preprocessed Data
        new_ds = ds[['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'combo']]
        with open('preprocess_data.pkl', 'wb') as file:
            pickle.dump(new_ds, file)
    
    # Initialize Model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("Model initialized successfully")
    return True

# Suggest Sections Function
def suggest_sections(complaint, dataset, min_suggestions=5):
    preprocessed_complaint = preprocess_text(complaint)
    complaint_embedding = model.encode(preprocessed_complaint)
    section_embedding = model.encode(dataset['combo'].tolist())
    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]
    similarity_threshold = 0.2
    relevant_indices = []

    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05

    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    suggestions = dataset.iloc[sorted_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court']].to_dict(orient='records')
    
    return suggestions

@app.route('/api/suggest', methods=['POST'])
def get_suggestions():
    try:
        data = request.get_json()
        complaint = data.get('complaint', '')
        
        if not complaint.strip():
            return jsonify({'error': 'Complaint text is required'}), 400
        
        if model is None or new_ds is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        suggestions = suggest_sections(complaint, new_ds)
        
        return jsonify(suggestions)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Loading model and data...")
    if load_model_and_data():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model and data. Exiting...") 