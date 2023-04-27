import torch
import torch.nn as nn
import transformers

#DistillBERT Model
class DistillBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.custom_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, input_ids, attention_mask):
        distillbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = distillbert_output.last_hidden_state[:, 0, :]
        
        output = self.custom_layer(last_hidden_state)
        
        return output



#RoBERTa mdoel
class RoBERTaClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base')
        self.custom_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = roberta_output.last_hidden_state[:, 0, :]
        
        output = self.custom_layer(last_hidden_state)
        
        return output


#BERT model
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.custom_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state[:, 0, :]
        
        output = self.custom_layer(last_hidden_state)
        
        return output



