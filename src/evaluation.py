import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score , precision_score , recall_score
import torch
from sklearn.metrics import precision_recall_fscore_support
import logging
from transformers import AutoModelForSequenceClassification , AutoTokenizer

import torch
from sklearn.metrics import precision_recall_fscore_support
import logging
from process import preprocess


def calculate_metrics(true_labels,predicted_labels):
    # Calculate accuracy for each label
    label_accuracies = []
    for i in range(len(true_labels)):
        label_accuracy = accuracy_score(true_labels[i], predicted_labels[i])
        label_accuracies.append(label_accuracy)

    label_precisions = []
    for i in range(len(true_labels)):
        label_prec = precision_score(true_labels[i], predicted_labels[i],average='macro',zero_division=True)
        label_precisions.append(label_prec)

    label_recalls = []
    for i in range(len(true_labels)):
        label_recall = recall_score(true_labels[i], predicted_labels[i],average='macro',zero_division=True)
        label_recalls.append(label_recall)

    # Calculate macro F1 score for each label and take the average
    label_macro_f1_scores = []
    for i in range(len(true_labels)):
        label_macro_f1 = f1_score(true_labels[i], predicted_labels[i], average='macro',zero_division=True)
        label_macro_f1_scores.append(label_macro_f1)

    # Calculate the average accuracy and macro F1 score
    average_accuracy = np.mean(label_accuracies)
    average_macro_f1 = np.mean(label_macro_f1_scores)
    average_precision = np.mean(label_precisions)
    average_recall = np.mean(label_recalls)
    return {
        "Accuracy":average_accuracy,
        "F1-Score":average_macro_f1,
        "Precision":average_precision,
        "Recall":average_recall        
    }

def ae_predict(test_file, model, tokenizer, logger=None):
    device = torch.device('cuda')
    targets = ["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"]

    model.eval()
    model = model.to(device)

    true_labels = []
    predicted_labels = []

    for j in range(len(test_file)-2800):
        data_point = test_file.iloc[j]
        text = data_point["text"]
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids,mask).logits

        probabilities = torch.sigmoid(outputs)
        threshold = 0.15
        predictions = (probabilities > threshold).cpu().numpy().tolist()[0]
        true_labels = [int(data_point[emotion]) for emotion in targets]
        
        predicted_labels.append(predictions)
        true_labels.append(true_labels)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    metrics = calculate_metrics(true_labels,predicted_labels)
    print(predicted_labels)
    print(true_labels)
    print(probabilities)
    if logger is not None:
        logger.info("Macro-Averaged Metrics:")
        logger.info(f"Precision={metrics['Precision']}, Recall={metrics['Recall']}, F1={metrics['F1-Score']}")

    model = model.to(torch.device('cpu'))
    return metrics


logging.basicConfig(level=logging.INFO,filename='logs/evaluation.log')
logger = logging.getLogger(__name__)
test_file = preprocess(pd.read_csv('final_datasets/emopars_test.csv'))

model = AutoModelForSequenceClassification.from_pretrained('HooshVareLab/bert-base-parsbert-uncased',num_labels=6)
model.load_state_dict(torch.load('checkpoints/emopars_parsbert.pth'))
tokenizer = AutoTokenizer.from_pretrained('HooshVareLab/bert-base-parsbert-uncased')
results = ae_predict(test_file, model, tokenizer, logger=logger)
print(results)