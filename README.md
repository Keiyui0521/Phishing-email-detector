# Phishing Email Detector

This project is the final group project for STAT1016 (Data Science 101) at The University of Hong Kong.

A Flask web application for detecting phishing emails using machine learning.

## Features

- Email submission form for phishing detection
- Machine learning model integration with confidence scores
- Error reporting system for false positives/negatives
- Admin panel for model management and retraining
- Security measures including input sanitization, rate limiting, and encryption

## Installation
1. Create virtual environment:
```bash
python3 -m venv venv_classifier
source venv_classifier/bin/activate
```

2. Install requirements:
```bash
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

## Project Structure

```
.
├── app.py                 # Main Flask application
├── data/                  # Directory for database and model versions
│   └── reports.db         # SQLite database for error reports (created on first run)
├── linear_model.sav       # Initial trained model
├── for ku.ipynb          # Original notebook with model training code
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates
│   ├── admin.html         # Admin panel template
│   ├── base.html          # Base template with layout
│   └── index.html         # Main page template
```

## Setup Instructions

1. Install dependencies:

```bash
pip3 install -r requirements.txt
```

2. Download the required spaCy model:

```bash
python3 -m spacy download en_core_web_sm
```

3. Run the application:

```bash
python app.py
```

4. Access the application at http://127.0.0.1:5000

## Usage

### User Interface

1. Paste an email into the text area on the home page
2. Click "Analyze Email" to get the prediction
3. View the result with confidence score
4. If the prediction is incorrect, click "Report Error" to submit a correction

### Admin Panel

1. Access the admin panel at http://127.0.0.1:5000/admin
2. Default credentials: username `admin`, password `admin123`
3. View error reports and model information
4. Retrain the model with collected error reports

## Security Features

- HTML input sanitization to prevent XSS attacks
- Rate limiting on API endpoints to prevent abuse
- Email content encryption in the database
- Admin authentication for protected endpoints

## Model Information

The phishing detection model uses a logistic regression classifier trained on email text data. The model preprocesses emails by:

1. Cleaning and normalizing text
2. Removing special characters and numbers
3. Converting to lowercase
4. Vectorizing using bag-of-words representation

The model maintains versioning to track improvements over time as more error reports are collected and used for retraining.