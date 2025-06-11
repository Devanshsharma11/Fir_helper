from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback

# Set up environment for Render
os.environ['NLTK_DATA'] = '/tmp/nltk_data'

app = Flask(__name__)
CORS(app)

# Download NLTK data for deployment
try:
    import nltk
    nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Import after NLTK setup
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    import pickle
    print("All imports successful")
except Exception as e:
    print(f"Error importing libraries: {e}")
    traceback.print_exc()

# Global variables
vectorizer = None
new_ds = None

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
        return text.lower()

def load_model_and_data():
    global vectorizer, new_ds
    
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        
        if not os.path.exists('preprocess_data.pkl'):
            print("ERROR: preprocess_data.pkl file not found!")
            return False
            
        new_ds = pickle.load(open('preprocess_data.pkl', 'rb'))
        print("Data loaded successfully")
        print(f"Dataset shape: {new_ds.shape}")
        
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        vectorizer.fit(new_ds['combo'].tolist())
        print("TF-IDF Vectorizer initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error loading model and data: {str(e)}")
        traceback.print_exc()
        return False

def suggest_sections(complaint, dataset, min_suggestions=5):
    try:
        preprocessed_complaint = preprocess_text(complaint)
        print(f"Preprocessed complaint: {preprocessed_complaint}")
        
        complaint_vector = vectorizer.transform([preprocessed_complaint])
        dataset_vectors = vectorizer.transform(dataset['combo'].tolist())
        
        similarities = cosine_similarity(complaint_vector, dataset_vectors)[0]
        top_indices = similarities.argsort()[-min_suggestions:][::-1]
        
        similarity_threshold = 0.1
        relevant_indices = [i for i in top_indices if similarities[i] > similarity_threshold]
        
        while len(relevant_indices) < min_suggestions and similarity_threshold > 0.01:
            similarity_threshold -= 0.01
            relevant_indices = [i for i in top_indices if similarities[i] > similarity_threshold]
        
        suggestions = dataset.iloc[relevant_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court']].to_dict(orient='records')
        return suggestions
        
    except Exception as e:
        print(f"Error in suggest_sections: {str(e)}")
        traceback.print_exc()
        return []

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'FIR Legal Section Recommender API (Render NLP Version)',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'suggest': '/api/suggest (POST)'
        },
        'model_loaded': vectorizer is not None,
        'data_loaded': new_ds is not None
    })

@app.route('/api/suggest', methods=['POST'])
def get_suggestions():
    try:
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
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': vectorizer is not None,
        'data_loaded': new_ds is not None,
        'data_shape': new_ds.shape if new_ds is not None else None,
        'version': 'Render NLP Version'
    })

if __name__ == '__main__':
    print("Starting Render NLP Backend...")
    print(f"Python version: {sys.version}")
    
    if load_model_and_data():
        print("Starting Flask server...")
        port = int(os.environ.get('PORT', 5001))
        print(f"Starting server on port: {port}")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load model and data. Exiting...")
        sys.exit(1) 