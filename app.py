from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./best_model")
tokenizer = BertTokenizer.from_pretrained("./best_model")
model.eval()

def predict_sentiment(sentence):
    inputs = tokenizer(
        sentence,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return label_mapping[predicted_class]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "Please enter a sentence."}), 400
    sentiment = predict_sentiment(sentence)
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)