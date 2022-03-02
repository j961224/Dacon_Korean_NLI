import pandas as pd
import collections
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer)

dict_label_to_num = {"contradiction":0, "entailment":1, "neutral":2}

def label_to_num(label):
    num_label = []
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label


class ExplainableModel_Dataset(Dataset):

  def __init__(self, dataset, model_path, max_length: int = 512):
      super().__init__()
      self.max_length = max_length
      self.result = []
      dataset = dataset.reset_index()
      for i in range(len(dataset)):
        self.result.append((dataset['premise'][i], dataset['hypothesis'][i], dataset['label'][i]))
      if len(self.result)==6913:
        self.result = self.result[:-1]
        print(len(self.result))
      self.tokenizer = AutoTokenizer.from_pretrained(model_path)

  def __len__(self):
      return len(self.result)

  def __getitem__(self, idx):
      premise, hypothesis, label = self.result[idx]
      print(premise)
      premise_input_ids = self.tokenizer.encode(premise, add_special_tokens=False)
      hypothesis_input_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
      input_ids = premise_input_ids + [2] + hypothesis_input_ids
      if len(input_ids) > self.max_length - 2:
          input_ids = input_ids[:self.max_length - 2]
      # convert list to tensor
      length = torch.LongTensor([len(input_ids) + 2])
      input_ids = torch.LongTensor([0] + input_ids + [2])
      label = torch.LongTensor([label])
      return input_ids, label, length


class NLI_Dataset(Dataset):
  def __init__(self, dataset, tokenizer, is_inference=False):
    self.premise =  dataset['premise']
    self.hypothesis = dataset['hypothesis']
    self.label = dataset['label']
    self.is_inference = is_inference
    self.tokenizer = tokenizer

  def __getitem__(self, idx):
    premise, hypothesis, label = self.premise[idx], self.hypothesis[idx], self.label[idx]

    if "roberta" in self.tokenizer.name_or_path and not "xlm" in self.tokenizer.name_or_path:
      tokenized_sentences = self.tokenizer(premise, hypothesis, truncation=True, padding=True, \
      max_length=256,  add_special_tokens=True, return_token_type_ids = False, return_tensors="pt" if self.is_inference else None)

    else:
      tokenized_sentences = self.tokenizer(premise, hypothesis, truncation=True, padding=True, \
      max_length=256,  add_special_tokens=True, return_tensors="pt" if self.is_inference else None)
    
    item = {
      key: val for key, val in tokenized_sentences.items()
    }
    item['label'] = torch.LongTensor([label])
    return item

  def __len__(self):
    return len(self.premise)



def preprocessing_dataset(dataset, sen_preprocessor):

    # Sentence Preprocessor
    premise = [sen_preprocessor(sen) for sen in dataset["premise"]]
    hypothesis = [sen_preprocessor(sub) for sub in dataset["hypothesis"]]

    out_dataset = pd.DataFrame(
        {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": dataset["label"],
        }
    )
    return out_dataset



def load_data(dataset_dir, k_fold=0, val_ratio=0, train=True):    
  
  dataset = pd.read_csv(dataset_dir)
  dataset = dataset.dropna()
  dataset = dataset.reset_index()
  if train:
    dataset['label'] = label_to_num(dataset["label"].values)

  if k_fold > 0 and train == True:  # train, split by kfold
      return split_by_kfolds(dataset, k_fold)
  elif val_ratio > 0:  # train, split by val_ratio
      return split_by_val_ratio(dataset, val_ratio)
  elif train == False:  # inference
      return dataset
  else:  # train, not split
      return [[dataset, None]]


def split_by_kfolds(dataset, k_fold):
    X = dataset.drop(["label"], axis=1)
    y = dataset["label"]
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    return [
        [dataset.iloc[train_dset], dataset.iloc[val_dset]]
        for train_dset, val_dset in skf.split(X, y)
    ]


def split_by_val_ratio(dataset, val_ratio):
    data_size = len(dataset)
    index_map = collections.defaultdict(list)
    for idx in range(data_size):
        label = dataset.iloc[idx]["label"]
        index_map[label].append(idx)
    train_indices = []
    val_indices = []

    for label in index_map.keys():
        idx_list = index_map[label]
        val_size = int(len(idx_list) * val_ratio)

        val_index = random.sample(idx_list, val_size)
        train_index = list(set(idx_list) - set(val_index))

        train_indices.extend(train_index)
        val_indices.extend(val_index)

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    train_dset = dataset.iloc[train_indices]
    val_dset = dataset.iloc[val_indices]
    return [[train_dset, val_dset]]


def load_mlm_data(dataset_dir, Preprocessor):
    pd_dataset = pd.read_csv(dataset_dir)
    sentence = pd_dataset["premise"]
    sentence1 = pd_dataset["hypothesis"]

    total_sentence = []
    for i in range(len(sentence)):
      total_sentence.append(sentence[i]+"[SEP]"+sentence1[i]) 
    pd_dataset['total_sentence'] = total_sentence
    pd_dataset['label']=label_to_num(pd_dataset["label"].values)

    pd_dataset = pd_dataset.drop_duplicates(
            ["total_sentence"], keep="first"
        ).reset_index()  # 중복되는 문장 제거

    pd_dataset = pd.DataFrame(
        {
            "index": pd_dataset['index'],
            "sentence": [Preprocessor(sen) for sen in pd_dataset["total_sentence"]],
            "label": pd_dataset["label"],
        }
    )
    return pd_dataset
