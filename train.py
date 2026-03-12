import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Load dataset
train = pd.read_csv("dataset/train.tsv", sep="\t", header=None)

print(train.head())

# Keep only label and text columns
train = train[[1, 2]]
train.columns = ["label", "text"]

# Convert labels to binary (true/mostly-true as 1, others as 0)
train["label"] = train["label"].apply(lambda x: 1 if x in ["true", "mostly-true"] else 0)

# Load tokenizer from :contentReference[oaicite:0]{index=0}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Convert pandas dataframe to Hugging Face dataset
dataset = Dataset.from_pandas(train)

# Tokenize dataset
dataset = dataset.map(tokenize, batched=True)

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)

# Remove unnecessary columns
dataset = dataset.remove_columns(["text"])

# Convert to PyTorch tensors
dataset.set_format("torch")

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    logging_steps=50
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train model
trainer.train()

# Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")