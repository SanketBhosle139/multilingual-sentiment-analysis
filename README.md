# Multilingual Sentiment Analysis

A transformer-based sentiment analysis system for multilingual and Hinglish text, built using a fine-tuned BERT model and Flask.

## Overview

This project classifies text into three sentiment categories:

- **Negative**
- **Neutral**
- **Positive**

A fine-tuned BERT sequence-classification model performs the sentiment prediction, while a Flask web application provides an interactive interface for users to enter text and receive predictions in real time.

## Features

- Multilingual and Hinglish sentiment classification
- Three-class sentiment prediction
- Fine-tuned BERT model
- Flask web application
- Real-time sentiment prediction
- Model evaluation with classification reports
- Confusion matrix generation

## Technologies Used

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- BERT
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- HTML/CSS/JavaScript

## Project Structure

```text
multilingual-sentiment-analysis/
│
├── best_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── templates/
│   └── index.html
│
├── app.py
├── evaluate_model.py
├── data.txt
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## How It Works

The application processes the user's input through the following pipeline:

```text
User enters text
       ↓
BERT Tokenizer
       ↓
Fine-tuned BERT Model
       ↓
Classification Layer
       ↓
Sentiment Prediction
       ↓
Negative / Neutral / Positive
```

1. **Input:** The user enters a sentence through the Flask web interface.
2. **Tokenization:** The BERT tokenizer converts the text into tokens and model inputs.
3. **Model Inference:** The fine-tuned BERT model processes the tokenized input.
4. **Classification:** The model produces scores for the three sentiment classes.
5. **Prediction:** The class with the highest score is returned as the predicted sentiment.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SanketBhosle139/multilingual-sentiment-analysis.git
cd multilingual-sentiment-analysis
```

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Running the Web Application

Start the Flask application:

```powershell
python app.py
```

The Flask development server will start locally. Open the URL displayed in the terminal in your web browser.

## Model Evaluation

The `evaluate_model.py` script evaluates the trained model using the dataset stored in `data.txt`.

The evaluation includes:

- Classification report
- Confusion matrix

Run:

```powershell
python evaluate_model.py
```

The confusion matrix is saved as:

```text
confusion_matrix.png
```

## Sentiment Labels

| Class | Sentiment |
|------:|-----------|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

## Dataset

The project uses text data containing multilingual and Hinglish sentences with sentiment labels.

The dataset is stored in:

```text
data.txt
```

## Model

The trained BERT model and tokenizer are stored in the `best_model` directory.

The model uses a sequence-classification architecture to predict one of three sentiment classes:

- Negative
- Neutral
- Positive

The large `model.safetensors` file is managed using **Git Large File Storage (Git LFS)**.

## Flask API

The application exposes a `/predict` endpoint that accepts POST requests containing a sentence.

Example request:

```json
{
    "sentence": "This movie was really good!"
}
```

Example response:

```json
{
    "sentiment": "positive"
}
```

The root endpoint `/` serves the web interface.

## Project Highlights

- Transformer-based NLP using BERT
- Fine-tuned sequence classification model
- Multilingual/Hinglish text processing
- Flask-based web application
- Model evaluation using classification metrics
- Confusion matrix visualization
- Git LFS for large model storage

## Author

**Sanket Bhosle139**

[GitHub](https://github.com/SanketBhosle139)