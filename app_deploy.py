from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import os
import traceback
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download NLTK data for deployment
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Text Preprocessing Function
def preprocess_text(text):
    try:
        text = text.lower()
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        preprocessed_text = ' '.join(words)
        return preprocessed_text
    except Exception as e:
        print(f"Error in preprocess_text: {str(e)}")
        # Fallback to simple preprocessing
        return text.lower().replace('.', ' ').replace(',', ' ')

# Global variables for model and dataset
vectorizer = None
new_ds = None

def load_model_and_data():
    global vectorizer, new_ds
    
    try:
        # Check if file exists
        if not os.path.exists('preprocess_data.pkl'):
            print("ERROR: preprocess_data.pkl file not found!")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            return False
            
        # Load preprocessed data
        new_ds = pickle.load(open('preprocess_data.pkl', 'rb'))
        print("Preprocessed data loaded successfully")
        print(f"Dataset shape: {new_ds.shape}")
        print(f"Columns: {new_ds.columns.tolist()}")
        
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        # Fit the vectorizer on the combo text
        vectorizer.fit(new_ds['combo'].tolist())
        print("TF-IDF Vectorizer initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error loading model and data: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

# Suggest Sections Function using TF-IDF and Cosine Similarity
def suggest_sections(complaint, dataset, min_suggestions=5):
    try:
        preprocessed_complaint = preprocess_text(complaint)
        print(f"Preprocessed complaint: {preprocessed_complaint}")
        
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
        
    except Exception as e:
        print(f"Error in suggest_sections: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'FIR Legal Section Recommender API',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'suggest': '/api/suggest (POST)'
        },
        'model_loaded': vectorizer is not None,
        'data_loaded': new_ds is not None,
        'deployment_info': {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'files_present': os.listdir('.') if os.path.exists('.') else []
        }
    })

@app.route('/api/suggest', methods=['POST'])
def get_suggestions():
    try:
        print(f"Received request: {request.get_json()}")
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        complaint = data.get('complaint', '')
        
        if not complaint.strip():
            return jsonify({'error': 'Complaint text is required'}), 400
        
        if vectorizer is None or new_ds is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        print(f"Processing complaint: {complaint}")
        suggestions = suggest_sections(complaint, new_ds)
        
        if not suggestions:
            return jsonify({'error': 'No suggestions found'}), 404
        
        print(f"Returning {len(suggestions)} suggestions")
        return jsonify(suggestions)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': vectorizer is not None,
        'data_loaded': new_ds is not None,
        'data_shape': new_ds.shape if new_ds is not None else None,
        'deployment_info': {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'files_present': [f for f in os.listdir('.') if f.endswith('.pkl')] if os.path.exists('.') else []
        }
    })

if __name__ == '__main__':
    print("Loading model and data...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    if load_model_and_data():
        print("Starting Flask server...")
        # Use PORT environment variable for deployment platforms
        port = int(os.environ.get('PORT', 5001))
        print(f"Starting server on port: {port}")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load model and data. Exiting...")
        sys.exit(1) 