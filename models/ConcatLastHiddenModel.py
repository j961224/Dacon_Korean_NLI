import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.cuda.amp import autocast

class RobertaClassificationHead_FourPooling(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size*4, config.hidden_size*4)
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )

    self.dropout = nn.Dropout(classifier_dropout)
    self.out_proj = nn.Linear(config.hidden_size*4,config.num_labels)
    self.tanh = nn.Tanh()
    
  def forward(self,x):
    x = self.dropout(x)
    x = self.dense(x)
    x = self.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class ConcatLastFourPoolingModel(nn.Module):
  def __init__(self,model_name, config):
    super().__init__()
    # self.model = RobertaModel(config)
    self.model = AutoModel.from_pretrained(model_name, config=config)
    self.config = config
    self.num_labels = config.num_labels

    self.classifier = RobertaClassificationHead_FourPooling(config)

  @autocast()
  def forward(self, input_ids, attention_mask):
    all_hidden_states = self.model(input_ids = input_ids, attention_mask = attention_mask)[2]

    concatenate_pooling = torch.cat(
      (all_hidden_states[-1],all_hidden_states[-2],
      all_hidden_states[-3],all_hidden_states[-4]),-1
    ) # (batch, seq_len, hidden_size*4) -> 마지막 layer 4개 추출

    concatenate_pooling = concatenate_pooling[:,0] # (batch, hidden_size*4)

    logits = self.classifier(concatenate_pooling)

    return {'logits': logits}
