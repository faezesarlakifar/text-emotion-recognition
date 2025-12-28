import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import pandas as pd
import config

from transformers import AutoModel
from tqdm import tqdm
from dataset import *
from models import *
from process import preprocess_pipeline , preprocess
from train import Trainer
from sklearn.model_selection import train_test_split


log_file_path = config.LOG_FILE_PATH
logging.basicConfig(level=logging.INFO, filename=log_file_path)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    lr = config.LEARNING_RATE
    batch_size = config.BATCH_SIZE
    bert_model_name = config.MODEL_NAME
    hidden_size = config.HIDDEN_SIZE
    num_epochs = config.NUM_EPOCHS
    random_state = config.RANDOM_STATE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emotions = config.EMOTIONS
    test_size = config.TEST_SIZE
    column_names = config.COLUMN_NAMES

    logger.info(f"Learning Rate: {lr}, Batch Size: {batch_size}, Model Name: {bert_model_name}, Hidden Size: {hidden_size}")
    logger.info(f"Number of Epochs: {num_epochs}, Random State: {random_state}, Device: {device}")
    logger.info(f"Emotions: {emotions}, Test Size: {test_size}, Column Names: {column_names}")


    # emopars_train = pd.read_csv(config.EMOPARS_TRAIN_PATH)
    # emopars_test = pd.read_csv(config.EMOPARS_TEST_PATH)

    # emopars_train = preprocess(emopars_train)
    # emopars_test = preprocess(emopars_test)
    emopars = preprocess(pd.read_csv('final_datasets/emopars.csv').head(1000))
    emopars_train , emopars_test = train_test_split(emopars,test_size=test_size,random_state=random_state)
    # tqdm.pandas()
    # emopars_train['text'] = emopars_train['text'].progress_apply(preprocess_pipeline)
    # emopars_test['text'] = emopars_test['text'].progress_apply(preprocess_pipeline)


    # emopars_train.to_csv('datasets/cleaned_datasets/emopars_train.csv')
    # emopars_test.to_csv('datasets/cleaned_datasets/emopars_test.csv')

    emo_train_texts = emopars_train['text'].tolist()
    emo_train_labels = emopars_train[emotions].to_numpy()
    emo_test_texts = emopars_test['text'].tolist()
    emo_test_labels = emopars_test[emotions].to_numpy()


    emo_trainset = EmoPars(emo_train_texts,emo_train_labels)
    emo_testset = EmoPars(emo_test_texts,emo_test_labels)

    train_dataloader = torch.utils.data.DataLoader(emo_trainset,batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(emo_testset,batch_size=batch_size,shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(bert_model_name,num_labels=config.NUM_LABELS).to(device)
    criterion = config.LOSS_FUNCTION
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    trainer = Trainer(model,criterion,optimizer=optimizer,device=device)
    trainer(train_dataloader,test_dataloader,num_epochs)
