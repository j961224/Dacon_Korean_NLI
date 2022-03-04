import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torch.cuda.amp import autocast

class RobertaClassificationHead_LSTM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )

    self.dropout = nn.Dropout(classifier_dropout)
    self.out_proj = nn.Linear(config.hidden_size*2,config.num_labels)
    self.tanh = nn.Tanh()
    
  def forward(self,x):
    x = self.dropout(x)
    x = self.dense(x)
    x = self.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class LSTM_Model(nn.Module):
    def __init__(self, MODEL_NAME, model_config):
        super().__init__()

        self.model_config= model_config
        self.model= AutoModel.from_pretrained(MODEL_NAME, config= self.model_config)
        self.hidden_dim= self.model_config.hidden_size # roberta hidden dim = 1024

        self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                            batch_first= True, bidirectional= True)
        self.fc= RobertaClassificationHead_LSTM(model_config)

    @autocast()
    def forward(self, input_ids, attention_mask):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output= self.model(input_ids= input_ids, attention_mask= attention_mask)[0]

        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        hidden, (last_hidden, last_cell)= self.lstm(output)

        # (16, 1024) (batch, hidden_dim)
        cat_hidden= torch.cat((last_hidden[0], last_hidden[1]), dim= 1)
        logits= self.fc(cat_hidden)
        
        return {'logits': logits}
