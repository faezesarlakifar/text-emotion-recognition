import torch
import torch.nn as nn
import numpy as np
import config
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.metrics import accuracy_score , f1_score
from dataclasses import dataclass

import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

@dataclass
class TrainConfig:
    NUM_EPOCHS: int = 5

import torch
from torch import nn
from torch.nn.functional import cross_entropy


def f1_score(y_true, y_pred, average='macro'):
    """Calculates F1-score (weighted harmonic mean of precision and recall) for multi-class classification.

    Args:
        y_true (torch.Tensor): Ground-truth labels (one-hot encoded).
        y_pred (torch.Tensor): Predicted class probabilities.
        average (str, optional): Averaging strategy ('micro', 'macro', 'samples', or 'none').
            - 'micro': Calculate overall F1-score across all classes.
            - 'macro': Calculate F1-score for each class and average.
            - 'samples': Calculate F1-score for each sample and average.
            - 'none': Return F1-score for each class without averaging.
            Defaults to 'macro'.

    Returns:
        torch.Tensor: F1-score(s).
    """

    assert y_true.size() == y_pred.size()
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)

    # True positives (tp), false positives (fp), false negatives (fn) per class
    tp = torch.sum(y_pred * y_true, dim=0)
    fp = torch.sum(y_pred * (1 - y_true), dim=0)
    fn = torch.sum((1 - y_pred) * y_true, dim=0)

    # Precision, recall, F1-score per class
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    if average == 'micro':
        # Micro-average: F1-score across all classes (weighted by support)
        support = torch.sum(y_true, dim=0)
        f1 = (f1 * support).sum() / torch.sum(support)
    elif average == 'macro':
        # Macro-average: Average F1-score across classes
        f1 = torch.mean(f1)
    elif average == 'samples':
        # Sample-average: F1-score for each sample and average
        f1 = torch.mean(f1, dim=0)
    elif average == 'none':
        # Return F1-score for each class without averaging
        pass
    else:
        raise ValueError(f"Invalid average method: {average}")

    return f1



def dynamic_class_weighting(model, criterion, optimizer, num_classes, f1_weight=0.2, epsilon=1e-9):
    """Dynamically adjusts class weights in PyTorch loss function based on F1-score.

    Args:
        model (nn.Module): PyTorch model for classification.
        criterion (nn.Module): PyTorch loss function (e.g., nn.CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        num_classes (int): Number of classes in the classification task.
        f1_weight (float, optional): Weight for F1-score in loss weighting calculation. Defaults to 0.2.
        epsilon (float, optional): Small value added to avoid division by zero. Defaults to 1e-9.

    Returns:
        None
    """


    # Calculate F1-score for each class
    f1 = f1_score(batch_labels, predictions, average='macro')
    class_weights = 1 / (f1 + epsilon)  # Inversely proportional to F1

    # Normalize class weights
    class_weights = class_weights / class_weights.sum()

    # Create or update weight tensor (device agnostic)
    weight_tensor = torch.from_numpy(class_weights).float().to(outputs.device)

    # Set weights for criterion (if supported) or create custom weighted loss
    if hasattr(criterion, 'weight'):
        criterion.weight = weight_tensor
    else:
        def weighted_loss(outputs, target):
            pass



class Trainer:
    def __init__(self, model, criterion, optimizer, device=config.DEVICE,num_gpus=1,log_file_path=config.LOG_FILE_PATH):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_gpus = num_gpus
        if self.num_gpus > 1 and torch.cuda.is_available():
            self.model = nn.DataParallel(model)
        else:
            self.model = model

        self.logger = self.setup_logger(log_file_path)

    def setup_logger(self, log_file_path):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)  # 10 MB per file, keep 5 backup files
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log_info(self, message):
        self.logger.info(message)

    def to_device(self, data):
        if self.num_gpus > 1 and torch.cuda.is_available():
            return {key: value.to(self.device) for key, value in data.items()}
        else:
            return data.to(self.device)

    def train_step(self, data):
        self.model.train()
        batch = data
        inputs , attention_mask , targets = self.to_device(batch['input_ids']) , self.to_device(batch['attention_mask']) , self.to_device(batch['labels'])
        self.optimizer.zero_grad()

        outputs = self.model(inputs,attention_mask).logits

        loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluation_step(self, data):
        self.model.eval()
        batch = data
        inputs , attention_mask , targets = self.to_device(batch['input_ids']) , self.to_device(batch['attention_mask']) , self.to_device(batch['labels'])

        with torch.no_grad():
            outputs = self.model(inputs,attention_mask).logits

        predictions = torch.argmax(outputs, dim=1) if targets is not None else None

        targets_np = targets.cpu().numpy() if targets is not None else None
        predictions_np = predictions.cpu().numpy() if predictions is not None else None

        return targets_np, predictions_np, outputs.cpu().numpy()


    def evaluate(self, dataloader):
        total_loss = 0
        total_samples = 0

        for data in tqdm(dataloader):
            targets, predictions, outputs = self.evaluation_step(data)


            if outputs is not None:
                outputs_tensor = torch.from_numpy(outputs).to(self.device)  # Convert NumPy array to PyTorch tensor
                targets_tensor = torch.from_numpy(targets).to(self.device)
                total_loss += self.criterion(outputs_tensor, targets_tensor).item()
                total_samples += len(data)

        avg_loss = total_loss / total_samples if total_samples > 0 else None
        return avg_loss


    def __call__(self, train_dataloader, test_dataloader, num_epochs):
        for epoch in range(num_epochs):
            self.log_info(f"Epoch {epoch + 1}/{num_epochs}:")

            # Training
            self.model.train()
            total_train_loss = 0
            total_train_samples = 0

            for train_data in tqdm(train_dataloader):
                train_loss = self.train_step(train_data)
                total_train_loss += train_loss if train_loss is not None else 0
                total_train_samples += len(train_data)


            avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else None
            self.log_info(f"  Train Loss: {avg_train_loss}")
            torch.save(self.model.state_dict(),config.CHECKPOINT_PATH)
            # Testing
            avg_test_loss = self.evaluate(test_dataloader)
            self.log_info(f"  Test Loss: {avg_test_loss}")
            self.log_info("=" * 30)
