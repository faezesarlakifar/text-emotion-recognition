### Notice
After completing our research, we will provide comprehensive code examples in this repository, including the training process and implementation of various methods, to ensure reproducibility and facilitate further exploration in the field

# emotion-recognition
Emotion recognition in text is a fundamental aspect of natural language understanding, 
with significant applications in various domains

## Applications
 - mental health monitoring
 - customer feedback analysis
 - content recommendation
 - chatbots

## Emotions in our predictions
 - Anger
 - Disgust
 - Fear
 - Sadness
 - Happiness
 - Surprise
 - Other

## Datasets
 - EmoPars
 - ArmanEmo

## Model Explanation
Our approach for emotion recognition in text utilizes the powerful XLM-RoBERTa, which is a pre-trained transformer-based language model. We take advantage of its deep understanding of language by fine-tuning it on two diverse datasets: EmoPars and ArmanEmo. This allows our model to capture a wide range of emotions effectively.
One key aspect of our approach is the incorporation of a single Gated Recurrent Unit (GRU) strategically placed before the final fully connected layer. This integration plays a crucial role in enhancing our model's ability to capture contextual dependencies within the text. By considering the sequential nature of the input, the GRU layer enables our model to better understand the emotional nuances present in the text.
The result of this strategic integration is an improved F-score, which serves as a measure of the model's performance. By adding the GRU layer, our model demonstrates a significant enhancement in capturing intricate emotional patterns, resulting in a higher accuracy of emotion prediction.

## How do the prediction
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
## Our results
| Model Configuration                                | AV fscore | AE fscore | AE recall | AE precision | PE fscore | PE recall | PE precision |
|----------------------------------------------------|-----------|-----------|-----------|-------------|-----------|-----------|-------------|
| Fine-tuning XLM-RoBERTa (base) on EmoPars for AE prediction && Fine-tuning ParsBERT on ArmanEmo for PE prediction | 0.42      | 0.55      | 0.66      | 0.52        | 0.28      | 0.36      | 0.43        |
| Fine-tuning XLM-RoBERTa (base) + GRU on EmoPars for AE prediction && Fine-tuning XLM-RoBERTa (base) on ArmanEmo for PE prediction | 0.46      | 0.59      | 0.86      | 0.49        | 0.33      | 0.39      | 0.47        |
| Fine-tuning XLM-RoBERTa (large) + GRU on EmoPars for AE prediction && fine-tuning XLM-RoBERTa (large) + GRU on ArmanEmo for PE prediction| 0.49      | 0.62      | 0.73      | 0.58        | 0.35      | 0.41      | 0.47        |
| Fine-tuning XLM-RoBERTa (large) + GRU on EmoPars for AE prediction && Fine-tuning XLM-RoBERTa (large) on ArmanEmo for PE prediction| 0.50      | 0.62      | 0.73      | 0.58        | 0.37      | 0.42      | 0.49        |



## Acknowledgements

We would like to express our gratitude to the creators and contributors of the ArmanEmo and EmoPars datasets for their valuable work and making their datasets publicly available for research purposes. We acknowledge their efforts in collecting and annotating the data, which greatly contributed to the development of our model. 

- ArmanEmo Dataset: [Link to ArmanEmo Dataset](https://github.com/arman-rayan-sharif/arman-text-emotion)
- EmoPars Dataset: [Link to EmoPars Dataset](https://github.com/nazaninsbr/Persian-Emotion-Detection)
