import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pickle

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

# Load and Preprocess Data
try:
    ds = pd.read_csv('/Users/devanshsharma/Downloads/FIR-DATA.csv')
    print(ds.head(2))
except UnicodeDecodeError as err:
    print(f"Error: {err}")

ds.fillna("Not mentioned", inplace=True)
ds['combo'] = ds['Description'] + " " + ds['Offense']
ds['combo'] = ds['combo'].apply(preprocess_text)

# Save and Load Preprocessed Data
new_ds = ds[['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'combo']]
with open('preprocess_data.pkl', 'wb') as file:
    pickle.dump(new_ds, file)
new_ds = pickle.load(open('preprocess_data.pkl', 'rb'))

# Initialize Model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Suggest Sections Function
def suggest_sections(complaint, dataset, min_suggestions=5):
    preprocessed_complaint = preprocess_text(complaint)
    complaint_embedding = model.encode(preprocessed_complaint)#used to convert complaint into vector
    section_embedding = model.encode(dataset['combo'].tolist())#used to convert fir dataset into vector 
    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]#applying cos_sim algo.    
    similarity_threshold = 0.2#similarity thershold is 20% means data should be similar more the 20%
    relevant_indices = []

    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05  # Decrease threshold slowly

    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    suggestions = dataset.iloc[sorted_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'combo']].to_dict(orient='records')
    
    return suggestions#basically ye function hame sort karne me help kar raha hai to find max similarity wali fir

# Input and Display Suggestions
complaint = input("Enter Crime Description: ")
suggest_section = suggest_sections(complaint, new_ds)

if suggest_section:
    print("Suggested sections are:")
    for suggestion in suggest_section:
        print(f"Description: {suggestion['Description']}")
        print(f"Offense: {suggestion['Offense']}")
        print(f"Punishment: {suggestion['Punishment']}")
else:
    print("No relevant sections found.")
