from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
vectorizer = None
new_ds = None

def load_model_and_data():
    global vectorizer, new_ds
    
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
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    # Fit the vectorizer on the combo text
    vectorizer.fit(new_ds['combo'].tolist())
    print("TF-IDF Vectorizer initialized successfully")
    return True

# Suggest Sections Function using TF-IDF and Cosine Similarity
def suggest_sections(complaint, dataset, min_suggestions=5):
    preprocessed_complaint = preprocess_text(complaint)
    
    # Transform the complaint and dataset
    complaint_vector = vectorizer.transform([preprocessed_complaint])
    dataset_vectors = vectorizer.transform(dataset['combo'].tolist())
    
    # Calculate cosine similarities
    similarities = cosine_similarity(complaint_vector, dataset_vectors)[0]
    
    # Get top similar indices
    top_indices = similarities.argsort()[-min_suggestions:][::-1]
    
    # Filter by similarity threshold
    similarity_threshold = 0.1
    relevant_indices = [i for i in top_indices if similarities[i] > similarity_threshold]
    
    # If not enough results, lower threshold
    while len(relevant_indices) < min_suggestions and similarity_threshold > 0.01:
        similarity_threshold -= 0.01
        relevant_indices = [i for i in top_indices if similarities[i] > similarity_threshold]
    
    suggestions = dataset.iloc[relevant_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court']].to_dict(orient='records')
    
    return suggestions

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'FIR Legal Section Recommender API',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'suggest': '/api/suggest (POST)'
        },
        'frontend': 'http://localhost:5173',
        'model_loaded': vectorizer is not None
    })

@app.route('/api/suggest', methods=['POST'])
def get_suggestions():
    try:
        data = request.get_json()
        complaint = data.get('complaint', '')
        
        if not complaint.strip():
            return jsonify({'error': 'Complaint text is required'}), 400
        
        if vectorizer is None or new_ds is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        suggestions = suggest_sections(complaint, new_ds)
        
        return jsonify(suggestions)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': vectorizer is not None})

if __name__ == '__main__':
    print("Loading model and data...")
    if load_model_and_data():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to load model and data. Exiting...") 