import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Assuming your data is in a CSV file with columns 'ai' and 'human'
csv_file_path = 'your_dataset.csv'

# Load your dataset
import pandas as pd
df = pd.read_csv(csv_file_path)

# Combine 'ai' and 'human' columns into a single 'text' column
df['text'] = df['ai'].astype(str) + ' ' + df['human'].astype(str)

# Create a label column where 'AI' is 1 and 'Human' is 0
df['label'] = (df['ai'] == 'AI').astype(int)

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the training and testing texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Convert the labels to PyTorch tensors
train_labels = torch.tensor(train_labels.tolist())
test_labels = torch.tensor(test_labels.tolist())

# Create a PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Define training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Collect predictions and true labels for F1 score calculation
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    # Calculate accuracy, loss, and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    avg_loss = total_loss / len(train_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1:.4f}')

# Evaluation
model.eval()
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

all_preds = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# Calculate accuracy, F1 score, and loss for the test set
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%, Test F1 Score: {f1:.4f}')
model_save_path = 'distilbert_ai_classifier.pth'
model.save_pretrained(model_save_path)

# Save the tokenizer
tokenizer_save_path = 'distilbert_tokenizer'
tokenizer.save_pretrained(tokenizer_save_path)