# Import Libraries
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


df = pd.read_csv("fake_job_postings.csv")

# Get the dimensions of the Dataset
print("Dimensions of the Dataset (Rows, Columns):")
df.shape

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# Text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()
    return ""

text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    df[col] = df[col].apply(preprocess_text)

df["text"] = df[text_columns].apply(lambda x: " ".join(x), axis=1)

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["fraudulent"], test_size=0.2, random_state=42, stratify=df["fraudulent"]
) 

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")

# Tokenize dataset
def tokenize_data(texts, labels):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=512)
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": list(labels)
    })

    # Tokenize dataset
from datasets import Dataset # Import the Dataset class
def tokenize_data(texts, labels):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=512)
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": list(labels)
    })

# Import Libraries
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Import TrainingArguments, Trainer, and AutoModelForSequenceClassification from transformers
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification # Import AutoModelForSequenceClassification
from transformers import AutoConfig # Import AutoConfig
from datasets import Dataset # Import the Dataset class
from transformers import AutoTokenizer

# Training arguments (fine tuning)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# Load the configuration of the pre-trained model, with the desired number of labels
config = AutoConfig.from_pretrained("papluca/xlm-roberta-base-language-detection", num_labels=2)

# Define the model, using the loaded configuration
# ignore_mismatched_sizes=True to handle the shape mismatch
model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection", config=config, ignore_mismatched_sizes=True) # Define the model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")

# Tokenize dataset
def tokenize_data(texts, labels):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=512)
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": list(labels)
    })

# Create train and test datasets using the tokenize_data function
train_dataset = tokenize_data(train_texts, train_labels)
test_dataset = tokenize_data(test_texts, test_labels)

# Trainer setup
trainer = Trainer(
    model=model, # Pass the defined model to the Trainer
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train model
trainer.train() 
