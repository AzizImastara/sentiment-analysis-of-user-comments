from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import numpy as np

app = Flask(__name__)

# ==================== PREPROCESSING ====================

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


stopwords = {
    'yang', 'di', 'ke', 'dan', 'dari', 'ini', 'itu', 'untuk', 'dengan',
    'atau', 'karena', 'pada', 'jadi', 'sudah', 'belum', 'ada', 'tidak',
    'bukan', 'saya', 'kami', 'kita', 'mereka', 'dia', 'nya', 'akan',
    'dalam', 'jika', 'lagi', 'sebagai', 'oleh', 'bagi', 'tentang',
    'apa', 'mengapa', 'bagaimana', 'adalah', 'saat', 'hingga', 'tp', 'yg'
}


def stemming(word):
    prefixes = ['ber', 'ter', 'me', 'di', 'ke', 'se', 'per']
    suffixes = ['kan', 'an', 'lah', 'kah', 'nya']

    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix):]
            break
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    return word


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_emojis(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()

    tokens = text.split()
    tokens = [stemming(w) for w in tokens if w not in stopwords]

    return ' '.join(tokens)

# ==================== END PREPROCESSING ====================

# Load the trained SVM model with TF-IDF (saved with joblib)
model_path = os.path.join(os.path.dirname(__file__), 'svm_tfidf_tokopedia.pkl')
loaded_data = joblib.load(model_path)

# Check if it's a pipeline or tuple (model, vectorizer)
if hasattr(loaded_data, 'predict'):
    # It's a pipeline
    pipeline = loaded_data
    tfidf_vectorizer = None
    model = pipeline
else:
    # It's a tuple (model, vectorizer)
    model, tfidf_vectorizer = loaded_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text.strip():
            return jsonify({
                'success': False,
                'error': 'Teks tidak boleh kosong!'
            })
        
        # Preprocessing text before classification
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            return jsonify({
                'success': False,
                'error': 'Teks tidak valid setelah preprocessing!'
            })
        
        # Transform text and predict with probability
        if tfidf_vectorizer is None:
            # Pipeline: directly predict
            prediction = model.predict([cleaned_text])[0]
            # Try to get probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([cleaned_text])[0]
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function([cleaned_text])[0]
                # Convert decision function to pseudo-probability using sigmoid
                proba_pos = 1 / (1 + np.exp(-decision))
                proba = [1 - proba_pos, proba_pos]
            else:
                proba = [0.5, 0.5]
        else:
            # Separate model and vectorizer
            text_tfidf = tfidf_vectorizer.transform([cleaned_text])
            prediction = model.predict(text_tfidf)[0]
            # Try to get probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_tfidf)[0]
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(text_tfidf)[0]
                # Convert decision function to pseudo-probability using sigmoid
                proba_pos = 1 / (1 + np.exp(-decision))
                proba = [1 - proba_pos, proba_pos]
            else:
                proba = [0.5, 0.5]
        
        # Get percentages
        negatif_persen = round(proba[0] * 100, 2)
        positif_persen = round(proba[1] * 100, 2)
        
        # Determine sentiment label
        if prediction == 1:
            sentiment = 'Positif'
            color = 'success'
            persen = positif_persen
        else:
            sentiment = 'Negatif'
            color = 'danger'
            persen = negatif_persen
        
        return jsonify({
            'success': True,
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'color': color,
            'persen': persen,
            'positif_persen': positif_persen,
            'negatif_persen': negatif_persen
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
