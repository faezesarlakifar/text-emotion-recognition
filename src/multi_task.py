import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score

def data_balancer(df,num_samples):
    import pandas as pd
    import random

    # Load your dataset

    # Calculate the number of samples needed for each class to reach approximately 10,000 samples
    desired_samples_per_class = num_samples // len(df['emotion'].unique())

    # Create a list to store oversampled data for each class
    oversampled_data = []

    # Iterate over each class
    for emotion in df['emotion'].unique():
        # Calculate the number of samples needed for the current class
        samples_needed = desired_samples_per_class - len(df[df['emotion'] == emotion])
        
        # If there are more samples needed for the current class
        if samples_needed > 0:
            # Sample with replacement to oversample from the current class
            oversampled_samples = df[df['emotion'] == emotion].sample(n=samples_needed, replace=True, random_state=42)
            oversampled_data.append(oversampled_samples)
        # If there are more samples than needed for the current class
        elif samples_needed < 0:
            # Drop random samples from the current class
            random_drop_indices = df[df['emotion'] == emotion].sample(n=-samples_needed, random_state=42).index
            df = df.drop(random_drop_indices)

    # Concatenate the original dataset with the oversampled data for each class
    balanced_df = pd.concat([df] + oversampled_data)

    # Shuffle the dataframe to mix the samples
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure that the final dataset contains exactly 10,000 samples
    if len(balanced_df) < num_samples:
        # If there are less than 10,000 samples, add random samples from the original dataset
        remaining_samples_needed = num_samples - len(balanced_df)
        random_samples = df.sample(n=remaining_samples_needed, random_state=42)
        balanced_df = pd.concat([balanced_df, random_samples])
    elif len(balanced_df) > num_samples:
        # If there are more than 10,000 samples, randomly drop excess samples
        balanced_df = balanced_df.sample(n=num_samples, random_state=42)

    # Shuffle the dataframe again after adding or removing samples
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df



def seed_everything(seed=42):
    """Seed everything to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

MODEL_NAME = 'HooshVareLab/bert-base-parsbert-uncased'
MAX_LENGTH = 128
ARMAN_LABEL_DICT = {'ANGRY': 0, 'FEAR': 1,'HAPPY': 2, 'HATE': 3, 'SAD': 4, 'SURPRISE': 5,'OTHER': 6}
EMOTIONS = ['Anger','Fear','Happiness','Hatred','Sadness','Wonder']
BATCH_SIZE = 8
LR = 5e-5
NUM_EPOCHS = 10
seed_everything(42)

def preprocess(df):
    # Normalize emotion labels
    emotion_columns = ['Anger', 'Fear', 'Happiness', 'Hatred', 'Sadness', 'Wonder']

    for col in emotion_columns:
        df[col] = df[col] / df[col].max()  # Normalize to the range [0, 1]

    threshold = 0.4
    for col in emotion_columns:
        df[col] = df[col].apply(lambda x: 1 if x >= threshold else 0)

    return df

arman_train = pd.read_csv('final_datasets/arman_train.csv')
arman_test = pd.read_csv('final_datasets/arman_test.csv')
short_train = pd.read_csv('final_datasets/train_short.csv')
short_test = pd.read_csv('final_datasets/test_short.csv')
seyali_train = pd.read_csv('final_datasets/train_seyali.csv')
seyali_test = pd.read_csv('final_datasets/test_seyali.csv')
emopars = preprocess(pd.read_csv('final_datasets/emopars.csv'))
corona = pd.read_csv('final_datasets/corona.csv')

seyali_train['emotion'] = seyali_train['emotion'].map({"fear":"FEAR","joy":"HAPPY","anger":"ANGRY","sad":"SAD","surprise":"SURPRISE","disgust":"HATE"})
seyali_test['emotion'] = seyali_test['emotion'].map({"fear":"FEAR","joy":"HAPPY","anger":"ANGRY","sad":"SAD","surprise":"SURPRISE","disgust":"HATE"})


train = pd.concat([arman_train,short_train,short_test,seyali_train,seyali_test])
train = data_balancer(train,27000)
test = arman_test
X_train = train['text'].tolist()
y_train = train['emotion'].tolist()
X_test = test['text'].tolist()
y_test = test['emotion'].tolist()
emo_train , emo_test = train_test_split(emopars,test_size=0.1,random_state=42)
X_emoTrain = emo_train['text'].values.tolist()
X_emoTest = emo_test['text'].values.tolist()
y_emoTrain = emo_train[EMOTIONS].values.tolist()
y_emoTest = emo_test[EMOTIONS].values.tolist()


class ArmanEmo(torch.utils.data.Dataset):
    def __init__(self,texts,labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_length = MAX_LENGTH
        self.labels_dict = ARMAN_LABEL_DICT

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(ARMAN_LABEL_DICT[label],dtype=torch.long)
        }
        return inputs


class EmoPars(torch.utils.data.Dataset):
    def __init__(self,texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_length = MAX_LENGTH

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float32)  # Use the provided numeric label directly
        }

        return inputs


class MLP(nn.Module):
    def __init__(
            self,
            hidden_size : int,
            intermediate_size : int,
        ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate)
        up = self.up_proj(x)
        fuse = self.dropout(gate * up)
        outputs = self.down_proj(fuse)
        return outputs

class MultiTaskBert(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super().__init__()
        self.shared_backbone = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.shared_backbone.config.hidden_size
        self.mlp_block = MLP(hidden_size=hidden_size, intermediate_size=hidden_size * 4)
        self.task1_classifier = nn.Linear(hidden_size, num_classes1)
        self.task2_classifier = nn.Linear(hidden_size, num_classes2)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x1, mask1, x2, mask2):
        task1_embed = self.dropout(self.shared_backbone(x1, mask1).last_hidden_state)
        task2_embed = self.dropout(self.shared_backbone(x2, mask2).last_hidden_state)
        concat_embed = torch.cat([task1_embed, task2_embed], dim=1)
        mlp_output = self.layer_norm(self.mlp_block(concat_embed)) + concat_embed
        task1_embed, task2_embed = torch.chunk(mlp_output, 2, dim=1)
        task1_logits = self.task1_classifier(task1_embed[:,-1,:])
        task2_logits = self.task2_classifier(task2_embed[:,-1,:])
        return task1_logits, task2_logits


arman_train_dataset = ArmanEmo(X_train,y_train)
arman_test_dataset = ArmanEmo(X_test,y_test)
emopars_train_dataset = EmoPars(X_emoTrain,y_emoTrain)
emopars_test_dataset = EmoPars(X_emoTest,y_emoTest)

arman_dataloader = torch.utils.data.DataLoader(arman_train_dataset, batch_size=BATCH_SIZE//2, shuffle=True)
emopars_dataloader = torch.utils.data.DataLoader(emopars_train_dataset, batch_size=BATCH_SIZE//2, shuffle=True)
arman_test_dataloader = torch.utils.data.DataLoader(arman_test_dataset, batch_size=BATCH_SIZE//2, shuffle=True)
emopars_test_dataloader = torch.utils.data.DataLoader(emopars_test_dataset, batch_size=BATCH_SIZE//2, shuffle=True)



# Initialize your model
model = MultiTaskBert(num_classes1=len(ARMAN_LABEL_DICT), num_classes2=6)  # Assuming num_classes2 is 2 for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion1 = nn.CrossEntropyLoss()  # For classification task
criterion2 = nn.BCEWithLogitsLoss()  # For binary classification task

device = torch.device('cuda')
model.to(device)
import itertools

shorter_dataloader = min(len(arman_dataloader), len(emopars_dataloader))
# arman_cycle = itertools.cycle(arman_dataloader)
# emopars_cycle = itertools.cycle(emopars_dataloader)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()  # Set model to training mode
    total_loss_task1 = 0.0
    total_loss_task2 = 0.0
    idx = 0
    arman_dataloader = tqdm(arman_dataloader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    emopars_dataloader = tqdm(emopars_dataloader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    for batch_arman, batch_emopars in zip(arman_dataloader, emopars_dataloader):
        # Move data to device
        input_ids_arman = batch_arman['input_ids'].to(device)
        attention_mask_arman = batch_arman['attention_mask'].to(device)
        labels_arman = batch_arman['labels'].to(device)
        
        input_ids_emopars = batch_emopars['input_ids'].to(device)
        attention_mask_emopars = batch_emopars['attention_mask'].to(device)
        labels_emopars = batch_emopars['labels'].to(device)

        # Forward pass
        task1_logits, task2_logits = model(input_ids_arman, attention_mask_arman,
                                           input_ids_emopars, attention_mask_emopars)
        
        print(task1_logits.shape,task2_logits.shape,labels_arman.shape,labels_emopars.shape)
        # Calculate loss
        loss_task1 = criterion1(task1_logits, labels_arman)
        loss_task2 = criterion2(task2_logits, labels_emopars)
        
        # Total loss
        total_loss = loss_task1 + loss_task2
        if idx % 100 == 0:
            print("Loss 1:",loss_task1.item()," / Loss 2:",loss_task2.item())
        

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Accumulate total loss for logging
        total_loss_task1 += loss_task1.item()
        total_loss_task2 += loss_task2.item()
        idx+=1

    # Calculate average loss for the epoch
    avg_loss_task1 = total_loss_task1 / len(arman_dataloader)
    avg_loss_task2 = total_loss_task2 / len(emopars_dataloader)

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Task1 Loss: {avg_loss_task1}, Task2 Loss: {avg_loss_task2}')
    print("-"*80)
    test_predictions = []
    test_labels = []
    test_loss = 0.0
    model.eval()
    for idx, batch in enumerate(tqdm(arman_test_dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs , _ = model(input_ids, attention_mask , input_ids , attention_mask)
        loss = criterion1(outputs, labels)

        batch_predictions = torch.argmax(outputs, dim=1).cpu().tolist()
        test_predictions.extend(batch_predictions)
        test_labels.extend(labels.cpu().tolist())
        test_loss += loss.item()
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_fscore = f1_score(test_labels, test_predictions, average='macro')
    print("MonoClass Loss (Test):", test_loss / len(arman_test_dataloader))
    print("Test Accuracy:", test_accuracy)
    print("Test F-score:", test_fscore)


    print("-"*80)
    test_loss = 0.0
    for idx,batch in enumerate(tqdm(emopars_test_dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        _ , logits = model(input_ids, attention_mask,input_ids,attention_mask)
        loss = criterion2(logits, labels)
        test_loss+=loss.item() 
        
    print("Epoch:", epoch + 1, "MultiLabel Test loss:", test_loss/len(emopars_test_dataloader))
    print("-"*80)


torch.save(model.state_dict(), 'multi_task_model.pth')