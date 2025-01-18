import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib 
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime

from config import Config

app = Flask(__name__)

app.config.from_object(Config)
client = MongoClient(app.config['MONGO_URI'])

db = client['users']
users_collection = db['users']

journal_db = client['Journal']
journal_collection = journal_db['journal']


def clean_text(s):
    s = s.lower()
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    s = " ".join([word for word in s.split()
                if word not in stopwords.words('english')
                or word in ['not', 'can']])
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def augment_text(text):
    try:
        blob = TextBlob(text)
        translated = blob.translate(to='fr').translate(to='en')
        return str(translated)
    except Exception as e:
        return text

def predict_new_sentence(sentence, vectorizer, model):
    preprocessed_sentence = clean_text(sentence)
    augmented_sentence = augment_text(preprocessed_sentence)
    new_embeddings = vectorizer.transform([augmented_sentence])
    new_prediction = model.predict(new_embeddings)
    
    return new_prediction[0]

label_mapping = {
    0: 'Normal',
    1: 'Depression',
    2: 'Suicidal',
    3: 'Anxiety',
    4: 'Bipolar',
    5: 'Stress',
    6: 'Personality disorder'
}


def decode_prediction(prediction):
    return label_mapping.get(prediction, "Unknown")

def predict_new_sentence_with_label(sentence, vectorizer, model):
    prediction = predict_new_sentence(sentence, vectorizer, model)
    decoded_label = decode_prediction(prediction)
    
    return decoded_label



with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

vectorizer=joblib.load('tfidf_vectorizer.pkl')

test_sentences = [
    "I feel very sad and don't want to talk to anyone.",
    "Life is good, and I feel happy today.",
    "I can't stop thinking about ending my life."
]


# print("Predictions:")
# for sentence in test_sentences:
#     prediction_label = predict_new_sentence_with_label(sentence, vectorizer, model)
#     print(f"Input: {sentence}")
#     print(f"Prediction: {prediction_label}\n")


@app.route("/api/mode_tracking", methods=["POST"])
def mode_track():
    data = request.get_json()
    if not data:
        return jsonify({"Alert": "You didn't write anything!"}), 400

    email = data.get('email')
    journal_id = data.get('journal_id') 

    if not email or not journal_id:
        return jsonify({"Error": "'email' and 'journal_id' are required fields."}), 400

    user_data = journal_collection.find_one({"email": email})
    if not user_data:
        return jsonify({"Error": "Email not found"}), 404


    if journal_id:
        journal_entry = None
        for date_entry in user_data['journal']:
            for entry in date_entry['entries']:
                if entry['_id'] == journal_id:
                    journal_entry = entry
                    break
            if journal_entry:
                break

        if not journal_entry:
            return jsonify({"Error": "Journal entry not found"}), 404

        journal_content = journal_entry["content"]
        mode_prediction = predict_new_sentence_with_label(journal_content, vectorizer,model)

        journal_collection.update_one(
            {
                "email": email,
                "journal.entries._id": journal_id
            },
            {
                "$set": {
                    "journal.$[].entries.$[entry].prediction": mode_prediction
                }
            },
            array_filters=[{"entry._id": journal_id}]
        )

        users_collection.update_one(
            {'email': email},
            {'$set': {'current_mode': mode_prediction}}
        )

        return jsonify({
            "message": "Mode tracking updated successfully.",
            "journal_id": journal_id,
            "content": journal_content,
            "prediction": mode_prediction
        }), 200
    

if __name__ == '__main__':
    app.run(debug=True)
