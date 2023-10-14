from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import re
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
nltk.download('stopwords')


def remove_irrelevant(df):

    """
    remove text smaller than 15 chars or the ones that are not a sentence
    """
    mask = (df['clean_text'].str.len() > 15) & (df['clean_text'] != "not a sentence")
    filtered_df = df[mask]

    return filtered_df


label = 'Global Citizen'
drop_duplicates_for_train = True
ratio = 0.2 # ratio of test to the whole dataset. "same" for a balanced dataset
NUM_LABELS = 1
LR = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 32
NUM_FOLDS = 3 # cross val
MODEL_NAME = "bert-base-uncased"
PROJECT_SWEEP_NAME = label + " second - 08 - 10"
BASE_FOLDER = "/home/mitrasadat.mirshafie/JC_folder/hailey/global citizen/"

df_concat = pd.read_csv(BASE_FOLDER + "df_concat_Global Citizen.csv")
print('length is ', len(df_concat))
df_concat = df_concat.dropna(subset=[label, "clean_text"])
print('length is ', len(df_concat))
df_concat.reset_index(drop=True, inplace=True)

df_concat = remove_irrelevant(df_concat)


# Tokenize and format the data
def tokenize_data(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)

    return input_ids, attention_masks, labels


def balanced(training, label='Global Citizen', ratio=1):


    ones = training[training[label] == 1]
    zeros = training[training[label] == 0]

    n = int(1/ratio * len(ones))
    print(n)

    zeros = training[training[label] == 0].sample(n=n, random_state=42)

    print('length of zeros = ', len(zeros))
    print('length of ones = ', len(ones))

    # Concatenate the balanced samples
    df = pd.concat([zeros, ones])

    # Shuffle the dataset
    df = shuffle(df, random_state=42)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    return df


# ---- stats 
zeros = df_concat[df_concat[label] == 0]
ones = df_concat[df_concat[label] == 1]

print('zeros length = ', len(zeros))
print('ones length = ', len(ones))

import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('cuda!')
else:
    device = torch.device("cpu")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Apply cross-validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)



def train(config=None, checkpoint_dir=""):

    with wandb.init(config=config):
        config = wandb.config
      
        df=balanced(df_concat, ratio=config['ratio'])
    
        # Prepare your data------------------------------------------
        X = df['clean_text'].values
        y = df[label].values
    
        f1_list = []
        accuracy_list = []
        recall_list = []
        roc_auc_list = []
        tp_class_1_list = []
        tn_class_0_list = []
        avg_train_loss_list = []
        false_positives_list = []
        false_negative_list = []

        LR = config['learning_rate']
        NUM_EPOCHS = config['num_epochs']
        RATIO = config['ratio']
        BATCH_SIZE = config['batch_size']
  
  
        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
          print(f"Fold {fold + 1}/{NUM_FOLDS}")
  
          X_train, X_val = X[train_index], X[val_index]
          y_train, y_val = y[train_index], y[val_index]
  
          # model creation
          model = BertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                num_labels=NUM_LABELS)
          model.to(device)
  
          # Tokenize and format the training and validation data
          print('Tokenize and format the training and validation data')
          input_ids_train, attention_masks_train, y_train = tokenize_data(X_train, y_train)
          input_ids_val, attention_masks_val, y_val = tokenize_data(X_val, y_val)
  
          input_ids_train = input_ids_train.to(device)
          attention_masks_train = attention_masks_train.to(device)
          y_train = y_train.to(device)
  
          input_ids_val = input_ids_val.to(device)
          attention_masks_val = attention_masks_val.to(device)
          y_val = y_val.to(device)
  
          # Create DataLoader for training and validation data
          print('Create DataLoader for training and validation data')
          train_data = TensorDataset(input_ids_train, attention_masks_train, y_train)
          train_sampler = RandomSampler(train_data)
          train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
  
          val_data = TensorDataset(input_ids_val, attention_masks_val, y_val)
          val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
  
          # Define optimizer and loss function
          optimizer = AdamW(model.parameters(), lr=LR)
          loss_fn = torch.nn.BCEWithLogitsLoss()
  
  
          # Training loop
          for epoch in range(NUM_EPOCHS):
              model.train()
              total_loss = 0
  
              for batch in train_dataloader:
                  optimizer.zero_grad()
                  input_ids, attention_mask, labels = batch
                  outputs = model(input_ids, attention_mask=attention_mask)
                  logits = outputs.logits
                  labels = labels.unsqueeze(1).float()
                  loss = loss_fn(logits, labels)
                  total_loss += loss.item()
                  loss.backward()
                  optimizer.step()
  
              avg_train_loss = total_loss / len(train_dataloader)
              print(f'Epoch {epoch + 1}/{NUM_EPOCHS} - Average training loss: {avg_train_loss:.4f}')
              avg_train_loss_list.append(avg_train_loss)
            
          # Evaluation
          model.eval()
          val_preds = []
          val_labels = []
  
          with torch.no_grad():
              for batch in val_dataloader:
                  input_ids, attention_mask, labels = batch
                  outputs = model(input_ids, attention_mask=attention_mask)
                  logits = outputs.logits
                  val_preds.extend(logits.sigmoid().round().squeeze(1).tolist())
                  val_labels.extend(labels.tolist())

        # Confusion matrix
          cm = confusion_matrix(val_labels, val_preds)
  
          # Get True Positives (TP) for class 1
          tp_class_1 = cm[1, 1]
          # Get True Negatives (TN) for class 0
          tn_class_0 = cm[0, 0]

          # Extract false positives
          false_positives = cm[0][1]
          false_negative = cm[1][0]

          # Calculate metrics for this fold
          f1 = f1_score(val_labels, val_preds)
          accuracy = accuracy_score(val_labels, val_preds)
          recall = recall_score(val_labels, val_preds)
          roc_auc = roc_auc_score(val_labels, val_preds)
  
          f1_list.append(f1)
          accuracy_list.append(accuracy)
          recall_list.append(recall)
          roc_auc_list.append(roc_auc)
          tp_class_1_list.append(tp_class_1)
          tn_class_0_list.append(tn_class_0)
          false_positives_list.append(false_positives)
          false_negative_list.append(false_negative)

          print(f'Fold {fold + 1} Metrics:')
          print(f'F1 Score: {f1:.4f}')
          print(f'Accuracy: {accuracy:.4f}')
          print(f'Recall: {recall:.4f}')
          print(f'ROC AUC: {roc_auc:.4f}')
  

  
          plt.figure(figsize=(8, 6))
          sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
          plt.xlabel('Predicted')
          plt.ylabel('True')
          plt.title('Confusion Matrix')
          plt.show()



        metrics_dict = {'tp_class_1': tp_class_1_list, 
                   'tn_class_0': tn_class_0_list,
                   'false_negative': false_negative_list,
                   'false_positives': false_positives_list,
                    'roc_auc': roc_auc_list, 
                    'recall': recall_list, 
                    'f1': f1_list,
                    'valid_accuracy':accuracy_list, 
                    'training_loss':avg_train_loss_list,
                    #'valid_loss':validation_loss, 'training_acc':top1_acc_train, 
                    }
        metrics_dict = {k: sum(v)/NUM_FOLDS for (k, v) in metrics_dict.items()}

        wandb.log(metrics_dict)
        print("Finished Training")
        #test_acc(model, device, config)
    
  
  

# HYPERPARAMETER SPACE DEFINITION---------------------------------------------

# Refer for distributions: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration

sweep_config = {

    # define the search method
    # one of "grid", "random" or "bayes"
    'method': 'bayes',

    # define the metric (useful for bayesian sweeps)
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    }
}

parameters = {
    # defining constant parameters
    # 'dataset': {'value': 'SetFit/sst2'},

    # define different types of losses for contrastive learning
    # these losses comes from sentence_transformers library

    'ratio': {
        'distribution': 'categorical',
        'values': [0.1, 0.3, 0.6, 1]
    },

    'batch_size': {
        # integers between 4 and 64
        'distribution': 'categorical',
        'values': [8, 16, 32, 64, 128, 256]
    },
    'num_epochs': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 4
    },

    'learning_rate': {
        'distribution': 'categorical',
        'values': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3],
    }
}

# adding the hyperparameters to the parameters field in the sweep_config dictionary
sweep_config['parameters'] = parameters
sweep_config


modelname = 'BERT'
modelpath = 'saved_checkpoint_' + modelname


sweep_id = wandb.sweep(sweep_config, project=PROJECT_SWEEP_NAME)


wandb.agent(sweep_id, train)