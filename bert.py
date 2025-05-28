import pandas as pd  # to load and read the file
import torch  # pytorch
import numpy as np  # to convert to numpy arrays
from torch.utils.data import DataLoader, Dataset  # to load data
# bert models to tokenize them
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # optimizer
# to split the data into validation and training
from sklearn.model_selection import train_test_split
# getting all the metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import optuna  # optuna can be helpful to try multiple parameters
from tqdm import tqdm  # to plot bars while training

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("train(3).csv")
df = df.dropna(subset=["Text", "Category"])
df['Category'] = df['Category'].astype('category').cat.codes

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset class


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(  # converts the text into embedings
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Train-validation split # splits data into 2 parts 20% and 80% validation and training
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Text'].tolist(), df['Category'].tolist(), test_size=0.2, random_state=42)

# data set is used for training
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
# dataset is used for testing
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

# Objective function for Optuna


def objective(trial):  # describing a parameters values for optuna
    # using small set of diffrent parameter because there are some issues on google collab and had to train on my laptop
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    num_epochs = trial.suggest_int("epochs", 2, 3)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)  # training data set
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size)  # testing dataset

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=43)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)  # optimizer

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device)
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluation the model on every set of parameter values
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device)
                      # this will move all tensors to device except labels
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)  # move labels to device
            outputs = model(**inputs)
            # pick the predicted label
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return f1_score(all_labels, all_preds, average="weighted")


# Run Optuna
study = optuna.create_study(direction="maximize")
# setting no of trails very low and i am running on my laptop
study.optimize(objective, n_trials=3)

# Best result
print("Best Hyperparameters:", study.best_trial.params)

# Retrain using best hyperparameters
best_params = study.best_trial.params
print("Training final model with best parameters:", best_params)

# Rebuild dataloaders with best batch size
train_loader = DataLoader(
    train_dataset, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])

# Load new model
final_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=43)
final_model.to(device)
# setting optimizer with best parameters
optimizer = AdamW(final_model.parameters(), lr=best_params['lr'])

# storing all required metrices
train_losses = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []
# training the model again with best parameters
for epoch in range(best_params['epochs']):
    final_model.train()
    running_train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"[Final Training] Epoch {epoch+1}"):
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = final_model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    train_losses.append(running_train_loss / len(train_loader))

    # evaluvating the Validation data
    final_model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device)
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = final_model(**inputs, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(accuracy_score(all_labels, all_preds))
    val_precisions.append(precision_score(
        all_labels, all_preds, average='weighted', zero_division=0))
    val_recalls.append(recall_score(all_labels, all_preds,
                       average='weighted', zero_division=0))
    val_f1s.append(f1_score(all_labels, all_preds, average='weighted'))

    print(f"\nEpoch {epoch+1} Metrics:")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Val Loss: {val_losses[-1]:.4f}")
    print(f"Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Precision: {val_precisions[-1]:.4f}")
    print(f"Recall: {val_recalls[-1]:.4f}")
    print(f"F1 Score: {val_f1s[-1]:.4f}")


# Save the final model
model_path = "bert_finetuned_model.pt"
torch.save(final_model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
# best parameter values are lr=3.04e-05;batch_size=16;epochs=3
