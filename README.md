[![AllerTrans](https://img.shields.io/badge/Publication-DOI:10.61186/itrc/16.3.35-red)](http://dx.doi.org/10.61186/itrc.16.3.35)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

<h1 align="center">
  EmoRecBiGRU
</h1>
<h2 align="center">
  Emotion Recognition in Persian Tweets with a Transformer-based Model, Enhanced by Bidirectional GRU
</h2>

## Abstract
Emotion recognition in text is a fundamental aspect of natural language understanding, with significant applications in various domains such as mental health monitoring, customer feedback analysis, content recommendation systems, and chatbots. In this paper, we present a hybrid model for predicting the presence of six emotions: anger, disgust, fear, sadness, happiness, and surprise in Persian text. We also predict the primary emotion in the given text, including these six emotions and the “other” category. Our approach involves the utilization of XLM-RoBERTa, a pre-trained transformer-based language model, and fine-tuning it on two diverse datasets: EmoPars and ArmanEmo. Central to our approach is the incorporation of a single Bidirectional Gated Recurrent Unit (BiGRU), placed before the final fully connected layer. This strategic integration enables our model to capture contextual dependencies more effectively, resulting in an improved F-score after incorporating the BiGRU layer. This enhanced model achieved a 2% improvement in the F-score metric on the ArmanEmo test set and a 7% improvement in the F-score metric for predicting the presence of six emotions on the final test set of the ParsiAzma Emotion Recognition competition.

## Goals

### 1. Predicting the Existence of All Emotions:
For each of the 6 classes, predict a binary label, meaning either the emotion is present in the given text or not. 
The emotions we consider are:
 - Anger
 - Disgust
 - Fear
 - Sadness
 - Happiness
 - Surprise

### 2. Predicting the Primary Emotion within 7 classes:
Predicting a single emotion that primarily represents the emotion of the given text within those 6 classes, along with an additional 'other' class for cases when the given text does not belong to one of those 6 classes.

## Datasets
The utilized dataset for this study includes two publicly available Datasets:
- ArmanEmo Dataset: [Link](https://github.com/arman-rayan-sharif/arman-text-emotion)
- EmoPars Dataset: [Link](https://github.com/nazaninsbr/Persian-Emotion-Detection)

## Model Architecture
<img src="https://github.com/faezesarlakifar/text-emotion-recognition/assets/63340593/dac0da99-fb4d-44a0-9a42-82f3bb545a25"
 alt="final model architecture" width="608" height="390">

## Hugging Face Model

### Model Checkpoints Availability
We have made our pre-trained model checkpoints available on Hugging Face. You can download the model checkpoints directly from the following links:
- [model_ae.pth](https://huggingface.co/sfaezella/Persian-EmoRoBERTa-BiGRU/resolve/main/model_ae.pth): The AE model checkpoint for All Emotion existence prediction.
- [model_pe.pt](https://huggingface.co/sfaezella/Persian-EmoRoBERTa-BiGRU/resolve/main/model_pe.pt): The PE model checkpoint for Primary Emotion recognition.

### Hugging Face Badge

To quickly access the model page, click the badge below:

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-green)](https://huggingface.co/sfaezella/Persian-EmoRoBERTa-BiGRU)

These models are ready for inference and easily loaded into your code.

## How to Make Predictions
```
AE_PATH = 'model_ae.pth'
PE_PATH = 'model_pe.pt'

checkpoint_ae = torch.load(AE_PATH, map_location=torch.device('cpu'))
model_ae.load_state_dict(checkpoint_ae)

checkpoint_pe = torch.load(PE_PATH, map_location=torch.device('cpu'))
model_pe.load_state_dict(checkpoint_pe)

pe_predict(test_file,model_pe,tokenizer)
pe_prediction = pe_predict(test_file,model_pe,tokenizer)
ae_prediction = ae_predict(test_file,model_ae,tokenizer)

ae_prediction['primary_emotion'] = pe_prediction['primary_emotion']
final_result.to_csv('final_result.csv')

```

## Results
### 1. ParsiAzma competition final results
### [link](https://parsiazma.ir/)
### 🏆 Second Place. =)
![Result Table](images/ParsiAzma-final-result.jpg)
  
## Acknowledgement
We want to thank the organizers of the ParsiAzma National Competition for providing the opportunity to conduct this research. Their dedication to studying and working in the emotion recognition area has been a driving force behind our project.

## Citation:
If you find our work valuable and it contributes to your research or projects, we kindly request that you use the following citation:
```
@ARTICLE{EmoRecBiGRU, 
author = {Sarlakifar, Faezeh and Mahdavi Mortazavi, Morteza and Shamsfard, Mehrnoush},  
title = {EmoRecBiGRU: Emotion Recognition in Persian Tweets with a Transformer-based Model, Enhanced by Bidirectional GRU}, 
volume = {16}, 
number = {3},  
journal = {International Journal of Information and Communication Technology Research},   
year = {2024}  
}
```
