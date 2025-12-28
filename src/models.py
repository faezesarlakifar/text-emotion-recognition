import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel,BertModel
from transformers import BertModel, BertTokenizer



class SBUNetwork(nn.Module):
    def __init__(self, num_classes,model_name,num_groups=4):
        super(SBUNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.gru = nn.GRU(self.bert.config.hidden_size, self.bert.config.hidden_size, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, bidirectional=True, batch_first=True)

        self.gru_group_norm = nn.GroupNorm(num_groups, self.bert.config.hidden_size * 2)
        self.lstm_group_norm = nn.GroupNorm(num_groups, self.bert.config.hidden_size * 2)
        self.gru_conv = nn.Conv1d(
            in_channels=self.bert.config.hidden_size * 2,
            out_channels=self.bert.config.hidden_size, 
            kernel_size=3,
            padding=1
        )

        self.lstm_conv = nn.Conv1d(
            in_channels=self.bert.config.hidden_size * 2,
            out_channels=self.bert.config.hidden_size, 
            kernel_size=3,
            padding=1
        )  
        self.gru_pool = nn.MaxPool1d(kernel_size=2,stride=2)
        self.lstm_pool = nn.MaxPool1d(kernel_size=2,stride=2)
        self.fc = nn.Linear(self.bert.config.hidden_size*2, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input_ids, attention_mask):
        # BERT embedding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = outputs.last_hidden_state

        # BiGRU Block        
        gru_output, _ = self.gru(bert_embedding)   
        gru_output = self.dropout(self.gru_group_norm(gru_output.permute(0,2,1)))        
        gru_conv_output = self.gru_conv(gru_output)  # Conv1d expects (batch_size, in_channels, seq_len) 
        gru_pooled_output = self.gru_pool(gru_conv_output)
        gru_pooled_output = gru_pooled_output.permute(0,2,1)[:,-1,:]  # Flatten the output

        # BiLSTM Block
        lstm_output, _ = self.lstm(bert_embedding)   
        lstm_output = self.dropout(self.lstm_group_norm(lstm_output.permute(0,2,1)) )       
        lstm_conv_output = self.lstm_conv(lstm_output)  # Conv1d expects (batch_size, in_channels, seq_len) 
        lstm_pooled_output = self.lstm_pool(lstm_conv_output)
        lstm_pooled_output = lstm_pooled_output.permute(0,2,1)[:,-1,:]  # Flatten the output
        
        concat_features = self.dropout(torch.concat((lstm_pooled_output,gru_pooled_output),dim=1))
        logits = self.fc(concat_features)
        return logits



# # # Example usage
# model_name = 'HooshVareLab/bert-base-parsbert-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = SBUNetwork(num_classes=2,model_name=model_name,num_groups=4)
# inputs = tokenizer("Hello, this is a test sentence.", return_tensors="pt", padding=True, truncation=True)
# output = model(inputs.input_ids, inputs.attention_mask)
# print(output)
