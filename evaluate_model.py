import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and tokenizer from the local directory
model = BertForSequenceClassification.from_pretrained("./best_model")
tokenizer = BertTokenizer.from_pretrained("./best_model")
model.eval()  # Set the model to evaluation mode

# Load the dataset
dataset = pd.read_csv("data.txt")  # Ensure the dataset is in the same folder
test_texts = dataset['Text'].tolist()  # Extract Hinglish sentences
test_labels = dataset['Label'].tolist()  # Extract sentiment labels (-1, 0, 1)

# Function to perform predictions
def get_predictions(model, tokenizer, test_texts):
    # Tokenize the test data
    inputs = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Perform predictions
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).numpy()  # Convert logits to class predictions
    return predictions

# Function to evaluate the model
def evaluate_model(predictions, true_labels):
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Negative", "Neutral", "Positive"]))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('confusion_matrix.png')  # Save the confusion matrix as an image
    plt.show()

# Perform predictions
predictions = get_predictions(model, tokenizer, test_texts)

# Evaluate the model
evaluate_model(predictions, test_labels)
