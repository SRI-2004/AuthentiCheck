import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def load_model_and_tokenizer(model_path, tokenizer_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def classify_text_with_threshold(model, tokenizer, text, threshold=0.99):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    print(probabilities)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    if probabilities[0][predicted_class].item() > threshold:
        return predicted_class
    else:
        return None


if __name__ == "__main__":
    # Replace with the actual paths where you saved your model and tokenizer
    model_path = 'distilbert_ai_classifier.pth'
    tokenizer_path = 'distilbert_tokenizer'

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    while True:
        user_input = input("Enter text to classify (type 'exit' to stop): ")

        if user_input.lower() == 'exit':
            break

        predicted_class = classify_text_with_threshold(model, tokenizer, user_input, threshold=0.8)

        if predicted_class is not None:
            if predicted_class == 1:
                print("Predicted: AI (Confidence > 0.8)")
            else:
                print("Predicted: Human (Confidence > 0.8)")
        else:
            print("Uncertain prediction (Confidence <= 0.8)")
