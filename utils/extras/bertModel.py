import os
import pandas as pd
import numpy as np
import json
import re
from nltk.tokenize import sent_tokenize 
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import transformers
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import datetime 
import warnings
warnings.filterwarnings('ignore')


max_training_tok = 16384 
def tokenize_and_label(text, tokenizer, max_length = max_training_tok):
    tokenized_inputs = tokenizer(
        text, 
        add_special_tokens=True, 
        truncation=True, 
        padding='max_length', 
        max_length=max_length, 
        return_tensors="pt"
    )
    # Tokenize the text
    tokenized_inputs = tokenizer(text, add_special_tokens=True, truncation=True, padding='max_length', max_length=max_training_tok, return_tensors="pt")
    tokenized_text = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])

    # Initialize label arrays
    section_labels = [-100 if token in ["[CLS]", "[SEP]", "[PAD]"] else 0 for token in tokenized_text]
    annotation_labels = [-100 if token in ["[CLS]", "[SEP]", "[PAD]"] else 0 for token in tokenized_text]


    # Helper function to assign labels based on spans
    def assign_labels(pattern, label_value, label_array):
        for match in re.finditer(pattern, text):
            start, end = match.span(1)  # Get span of the content inside the annotation
            for i, token in enumerate(tokenized_text):
                token_span = tokenized_inputs.token_to_chars(0, i)
                if token_span is not None:
                    token_start, token_end = token_span
                    # Check if the token is within the span of the annotation
                    if start <= token_start < end or start < token_end <= end:
                        label_array[i] = label_value

    # Assign labels for sections and annotations
    assign_labels(r'\{\#s(.*?)\/\}', 1, section_labels)
    assign_labels(r'\[\#a(.*?)\]', 1, annotation_labels)
    assign_labels(r'\[\#dt(.*?)\]', 2, annotation_labels)
    assign_labels(r'\[\#p(.*?)\]', 3, annotation_labels)

    return tokenized_text, section_labels, annotation_labels

def load_and_process_data(root_dir):
    all_annotated_data = []
    full_text = ""

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'privacy_policy.txt':
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    full_text += text + " "  # Append text from each file
                    sections = re.findall(r'\{\#s(.*?)\/\}', text, re.DOTALL)
                    for section in sections:
                        actions = re.findall(r'\[\#a(.*?)\]', section)
                        data_types = re.findall(r'\[\#dt(.*?)\]', section)
                        purposes = re.findall(r'\[\#p(.*?)\]', section)
                        all_annotated_data.append({'section': section, 'actions': actions, 'data_types': data_types, 'purposes': purposes})
    
    return full_text, all_annotated_data

# Function to remove consecutive special tokens
def remove_tokens(tokenized_text, token_section_labels, token_annotation_labels):
    # Create a list to store the indices of items to delete
    indices_to_delete = []
    tokens_to_drop = {'ns', 'nsc','[', ']', '#', '/', '}', '{'}
    # Iterate through the tokenized_text
    i = 0
    while i < len(tokenized_text):
        token = tokenized_text[i]

        # Check for consecutive special tokens and pattern
        if '[' in token and i + 2 < len(tokenized_text):
            next_tokens = tokenized_text[i + 1:i + 3]
            if '#' in next_tokens and next_tokens[-1] in ['a', 'dt', 'p']:
                indices_to_delete.extend([i, i + 1, i + 2])  # Delete the consecutive tokens
                i += 3  # Move to the token after the consecutive tokens
            else:
                i += 1  # Move to the next token
        elif '{' in token and i + 2 < len(tokenized_text):
            next_tokens = tokenized_text[i + 1:i + 3]
            if '#' in next_tokens and next_tokens[-1] == 's':
                indices_to_delete.extend([i, i + 1, i + 2])  # Delete the consecutive tokens
                i += 3  # Move to the token after the consecutive tokens
            else:
                i += 1  # Move to the next token
        elif token in tokens_to_drop:
            indices_to_delete.append(i)
            i += 1  # Move to the next token
        else:
            i += 1  # Move to the next token

    # Remove items from all three lists simultaneously
    for index in sorted(indices_to_delete, reverse=True):
        del tokenized_text[index]
        del token_section_labels[index]
        del token_annotation_labels[index]


tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

root_dir = 'policies_annotated/annotated_policies'

# Load and process data
privacy_policy_text, annotated_data = load_and_process_data(root_dir)

# Tokenize and label the data
tokenized_text, token_section_labels, token_annotation_labels = tokenize_and_label(privacy_policy_text, tokenizer)


# Remove consecutive special tokens
remove_tokens(tokenized_text, token_section_labels, token_annotation_labels)

# Pad tokenized text and labels to the seq length
max_length = 256

def pad_sequence(sequence, desired_length, tokenizer):
    pad_token_id = tokenizer.pad_token_id  # Get the special pad token ID from the tokenizer
    return sequence + [pad_token_id] * (desired_length - len(sequence))

tokenized_text = pad_sequence(tokenized_text, max_length, tokenizer)
token_section_labels = pad_sequence(token_section_labels, max_length, tokenizer)
token_annotation_labels = pad_sequence(token_annotation_labels, max_length, tokenizer)

def replace_integers_with_special_tag(tokenized_text, special_tag="[PAD]"):
    return [special_tag if isinstance(token, int) else token for token in tokenized_text]

# Replace any integer tokens with a special tag
tokenized_text = replace_integers_with_special_tag(tokenized_text)



# Print a portion of the processed data for review
for token, sec_label, ann_label in zip(tokenized_text, token_section_labels, token_annotation_labels):
    print(f"{token}: Section Label = {sec_label}, Annotation Label = {ann_label}")

print(len(privacy_policy_text))
print(len(tokenized_text), len(token_section_labels), len(token_annotation_labels))




class PrivacyPolicyDataset(Dataset):
    def __init__(self, input_ids, section_labels, annotation_labels):
        self.input_ids = input_ids
        self.section_labels = section_labels
        self.annotation_labels = annotation_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.input_ids[idx] != tokenizer.pad_token_id,
            'section_labels': self.section_labels[idx],
            'annotation_labels': self.annotation_labels[idx]
        }
# Tokenize and Label the Data (Assuming this step is already correctly done)
tokenized_text, token_section_labels, token_annotation_labels = tokenize_and_label(privacy_policy_text, tokenizer)

# Convert tokenized text to token IDs
input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

# Split the data into sequences
def split_into_sequences(data, sequence_length):
    return [data[i:i + sequence_length] for i in range(0, len(data), sequence_length)]

input_ids_split = split_into_sequences(input_ids, sequence_length=256)
section_labels_split = split_into_sequences(token_section_labels, sequence_length=256)
annotation_labels_split = split_into_sequences(token_annotation_labels, sequence_length=256)

# Convert lists to tensors
input_ids_tensor = torch.tensor(input_ids_split)
section_labels_tensor = torch.tensor(section_labels_split)
annotation_labels_tensor = torch.tensor(annotation_labels_split)

# Create dataset and dataloader
dataset = PrivacyPolicyDataset(input_ids_tensor, section_labels_tensor, annotation_labels_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Debugging: Check dataset and dataloader
print(f"Dataset size: {len(dataset)} sequences")
print(f"Total batches per epoch: {len(dataloader)}")

from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch.nn as nn

class BertForTokenAndSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_token_labels = 4  # Adjust this based on your needs

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Token classification head
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        
        # Sequence classification head
        self.seq_classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, labels=None, token_labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = self.dropout(outputs[0])
        pooled_output = self.dropout(outputs[1])

        # Token-level classification
        token_logits = self.token_classifier(sequence_output)

        # Sequence-level classification
        sequence_logits = self.seq_classifier(pooled_output)

        return token_logits, sequence_logits

# Load the configuration from 'bert-base-uncased'
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2  # Adjust based on your sequence classification needs

# Create an instance of your model
model = BertForTokenAndSequenceClassification(config)

# Load pre-trained BERT model

checkpoint_dir = './checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define optimizer and loss functions
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
token_loss_fn = nn.CrossEntropyLoss()
sequence_loss_fn = nn.CrossEntropyLoss()

num_epochs = 100
print(f"Dataset size: {len(dataset)} sequences")

# Check the DataLoader
print(f"Batch size: {dataloader.batch_size}")
print(f"Total batches per epoch: {len(dataloader)}")
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss_accumulated = 0  # To accumulate loss for reporting

    for batch in dataloader:
        optimizer.zero_grad()

        # Ensure tensors are on the same device as the model
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        token_labels = batch['annotation_labels'].to(model.device)
        section_labels = batch['section_labels'].to(model.device)

        # Forward pass
        token_logits, sequence_logits = model(input_ids, attention_mask=attention_mask)

        # Compute losses
        token_loss = token_loss_fn(
            token_logits.view(-1, model.num_token_labels), 
            token_labels.view(-1)
        )
        
        # Make sure section_labels is a 1D tensor with size [batch_size]
        if section_labels.dim() > 1:
            # Assuming the label for a sequence is the first element in section_labels
            section_labels = section_labels[:, 0]

        sequence_loss = sequence_loss_fn(
            sequence_logits.view(-1, model.num_labels), 
            section_labels
        )

        # Combine losses
        total_loss = token_loss + sequence_loss
        total_loss_accumulated += total_loss.item()

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Batch Loss: {total_loss.item()}")

    average_loss = total_loss_accumulated / len(dataloader)
    print(f"Epoch {epoch}, Average Loss: {average_loss}")

    #Rewrite each epoch 
    model_save_path = os.path.join(checkpoint_dir, 'berts1.0.model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
