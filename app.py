from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import joblib
import os
import logging
from datetime import datetime
import re # Added
import spacy # Added
# from sklearn.feature_extraction.text import CountVectorizer # Assuming model is a pipeline

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages

# Configure logging
logging.basicConfig(filename='error_reports.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# --- Text Cleaning (from Method.ipynb) ---
RE_PATTERNS = {
    ' are not ' : ["aren't"],
    ' cannot ' : ["can't"],
    ' cannot have ': ["can't've"],
    ' because ': ["cause"],
    ' could have ': ["could've"],
    ' could not ': ["couldn't"],
    ' could not have ': ["couldn't've"],
    ' did not ': ["didn't"],
    ' does not ': ["doesn't"],
    ' do not ': ["don't"],
    ' had not ': ["hadn't"],
    ' had not have ': ["hadn't've"],
    ' has not ': ["hasn't"],
    ' have not ': ["haven't"],
    ' he would ': ["he'd"],
    ' he would have ': ["he'd've"],
    ' he will ': ["he'll"],
    ' he is ': ["he's"],
    ' how did ': ["how'd"],
    ' how will ': ["how'll"],
    ' how is ': ["how's"],
    ' i would ': ["i'd"],
    ' i will ': ["i'll"],
    ' i am ': ["i'm"],
    ' i have ': ["i've"],
    ' is not ': ["isn't"],
    ' it would ': ["it'd"],
    ' it will ': ["it'll"],
    ' it is ': ["it's"],
    ' let us ': ["let's"],
    ' madam ': ["ma'am"],
    ' may not ': ["mayn't"],
    ' might have ': ["might've"],
    ' might not ': ["mightn't"],
    ' must have ': ["must've"],
    ' must not ': ["mustn't"],
    ' need not ': ["needn't"],
    ' ought not ': ["oughtn't"],
    ' shall not ': ["shan't"],
    ' shall not ': ["sha'n't"],
    ' she would ': ["she'd"],
    ' she will ': ["she'll"],
    ' she is ': ["she's"],
    ' should have ': ["should've"],
    ' should not ': ["shouldn't"],
    ' that would ': ["that'd"],
    ' that is ': ["that's"],
    ' there had ': ["there'd"],
    ' there is ': ["there's"],
    ' they would ': ["they'd"],
    ' they will ': ["they'll"],
    ' they are ': ["they're"],
    ' they have ': ["they've"],
    ' was not ':[ "wasn't"],
    ' we would ': ["we'd"],
    ' we will ': ["we'll"],
    ' we are ': ["we're"],
    ' we have ': ["we've"],
    ' were not ': ["weren't"],
    ' what will ': ["what'll"],
    ' what are ': ["what're"],
    ' what is ': ["what's"],
    ' what have ':[ "what've"],
    ' where did ': ["where'd"],
    ' where is ': ["where's"],
    ' who will ': ["who'll"],
    ' who is ': ["who's"],
    ' will not ': ["won't"],
    ' would not ': ["wouldn't"],
    ' you would ': ["you'd"],
    ' you will ': ["you'll"],
    ' you are ': ["you're"],
    ' american ':
        [
            'amerikan'
        ],
    ' though ': ['tho'],
    #' picture ': ['pic', 'pics'],
    ' soo ': ['so'],
    ' should ':['shoulda'],
    " aint ": ["am not"],
}

# Load spaCy model (consider error handling if model not downloaded)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # Depending on requirements, you might want to exit or handle this differently
    nlp = None # Set nlp to None if loading fails

def clean_text(text, remove_repeat_text=True, remove_patterns_text=True, is_lower=True):
    if remove_patterns_text:
        for target, patterns in RE_PATTERNS.items():
            for pat in patterns:
                text=str(text).replace(pat, target)

    if remove_repeat_text:
        text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Stopword removal was commented out in the notebook, keeping it commented here
    # if nlp:
    #     text = remove_stopwords(text, nlp) # Assuming remove_stopwords function exists if needed

    pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
    text = re.sub(pattern, '', text)
    text = str(text).replace("\n", " ")
    text = re.sub(r"http\S+", "", text)
    # Corrected regex for removing non-word/space chars, keeping spaces
    text = re.sub(r'[^\w\s]',' ',text) 
    text = re.sub('[0-9]',"",text)
    text = re.sub(" +", " ", text)
    text = re.sub("([^\x00-\x7F])+"," ",text)

    if is_lower:
        text=text.lower()

    return text.strip() # Added strip()

# --- Model Loading ---
MODEL_DIR = 'models' # Optional: Directory to store multiple model versions
DEFAULT_MODEL_NAME = 'linear_model.sav'
models = {}

def load_models():
    """Loads all .sav models from the root directory or MODEL_DIR."""
    global models
    models = {}
    potential_paths = [DEFAULT_MODEL_NAME] # Check root first
    if os.path.exists(MODEL_DIR):
        potential_paths.extend([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.sav')])

    loaded_default = False
    for model_path in potential_paths:
        model_name = os.path.basename(model_path)
        if os.path.exists(model_path):
            try:
                models[model_name] = joblib.load(model_path)
                print(f"Loaded model: {model_name} from {model_path}")
                if model_name == DEFAULT_MODEL_NAME:
                    loaded_default = True
            except Exception as e:
                print(f"Error loading model {model_name} from {model_path}: {e}")

    if not models:
        print("Warning: No models loaded. Prediction will not work.")
    elif not loaded_default:
        print(f"Warning: Default model '{DEFAULT_MODEL_NAME}' not found or failed to load.")

load_models() # Load models on startup

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main page with model selection."""
    available_models = list(models.keys())
    # Ensure default model is listed even if loading failed, but maybe disable selection?
    if DEFAULT_MODEL_NAME not in available_models and os.path.exists(DEFAULT_MODEL_NAME):
         available_models.insert(0, DEFAULT_MODEL_NAME) # Add it if file exists
    elif DEFAULT_MODEL_NAME not in available_models and os.path.exists(os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)):
         available_models.insert(0, DEFAULT_MODEL_NAME)

    # Determine a fallback model if the default isn't loaded
    current_default_model = DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in models else (available_models[0] if available_models else None)

    return render_template('index.html', models=available_models, selected_model=current_default_model)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles email prediction requests."""
    try:
        email_body = request.form['email_body']
        selected_model_name = request.form.get('model_version', DEFAULT_MODEL_NAME)

        if not email_body:
            flash('Please enter email text.', 'error')
            return redirect(url_for('index'))

        if selected_model_name not in models:
             flash(f'Selected model "{selected_model_name}" not found or failed to load. Cannot predict.', 'error')
             # Maybe redirect or show available models again?
             return redirect(url_for('index'))

        model = models[selected_model_name]

        # --- Preprocessing and Prediction --- 
        # 1. Clean the text using the function from the notebook
        cleaned_email = clean_text(email_body)

        # 2. Predict using the loaded model (assuming it's a Pipeline)
        # The model pipeline should handle vectorization (e.g., CountVectorizer) internally
        try:
            # Models often expect a list/iterable of documents
            prediction = model.predict([cleaned_email])[0]
            # Try to get probabilities for confidence score
            try:
                probabilities = model.predict_proba([cleaned_email])[0]
                # Assuming class 1 is Phishing, class 0 is Not Phishing
                confidence = probabilities[prediction] # Confidence of the predicted class
            except AttributeError:
                # Model might not have predict_proba (e.g., LinearSVC)
                confidence = 0.5 # Assign neutral confidence if unavailable
                print(f"Model {selected_model_name} does not support predict_proba.")

        except Exception as pred_e:
             print(f"Error during model prediction with {selected_model_name}: {pred_e}")
             flash(f'Error predicting with model {selected_model_name}. It might expect different input or preprocessing. Check logs.', 'error')
             return redirect(url_for('index'))

        # --- Format Result ---
        result_text = "Phishing" if prediction == 1 else "Not Phishing"
        confidence_score = confidence * 100

        return render_template('index.html',
                               prediction=result_text,
                               confidence=f"{confidence_score:.2f}%",
                               email_body=email_body,
                               models=list(models.keys()),
                               selected_model=selected_model_name)

    except Exception as e:
        print(f"Prediction route error: {e}")
        flash(f'An unexpected error occurred: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/report', methods=['POST'])
def report_error():
    """Logs user feedback about incorrect predictions."""
    try:
        email_body = request.form['email_body']
        reported_as = request.form['reported_as'] # 'phishing' or 'not_phishing'
        model_used = request.form['model_used']
        predicted_as = request.form['predicted_as']

        # Log to file
        log_message = f"REPORT - Model: {model_used}, Predicted: {predicted_as}, Reported Correct: {reported_as}, Email: '{email_body[:200]}...'" # Log snippet
        logging.info(log_message)
        print(f"Logged report: {log_message}")

        # Optionally, save to a database or structured file here
        # Example: Append to a CSV
        # with open('reports.csv', 'a', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     if os.path.getsize('reports.csv') == 0:
        #          writer.writerow(['timestamp', 'model_used', 'predicted_as', 'reported_as', 'email_snippet'])
        #     writer.writerow([datetime.now().isoformat(), model_used, predicted_as, reported_as, email_body[:200]])

        flash('Thank you for your feedback!', 'success')
    except Exception as e:
        print(f"Error reporting failed: {e}")
        flash(f'Error submitting report: {e}', 'error')

    return redirect(url_for('index'))

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory.")
        # Create a basic index.html if it doesn't exist (already created by previous step)
        # index_path = os.path.join('templates', 'index.html')
        # if not os.path.exists(index_path):
        #     with open(index_path, 'w') as f:
        #         f.write('<!DOCTYPE html><html><head><title>Phishing Detector</title></head><body><h1>Phishing Detector</h1><p>Template file created. Needs content.</p></body></html>')
        #     print("Created basic 'templates/index.html'.")

    # Check if spaCy model is loaded
    if nlp is None:
        print("\nERROR: spaCy model 'en_core_web_sm' could not be loaded.")
        print("Please run 'python -m spacy download en_core_web_sm' and restart the application.")
        # Optionally exit if spaCy is strictly required, though clean_text might partially work
        # exit(1)

    app.run(debug=True, host='0.0.0.0', port=5000) # Specify port 5000