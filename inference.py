from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,
)
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import importlib

import pickle as pickle
import numpy as np
import argparse
import os
from tqdm import tqdm
from functools import partial
from utils.tokenization import *
from utils.collate_functions import collate_to_max_length

from models.transformers_with_bilstm import LSTM_Model
from models.Explainable_Model import ExplainableModel


def inference(model, tokenized_sent, device, args, is_roberta=False):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()

    output_pred = []
    output_prob = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            if is_roberta:
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                )
            else:
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    token_type_ids=data["token_type_ids"].to(device),
                )
        if args.use_bilstm_model:
          logits = outputs['logits']
        else:
          logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def inference_ensemble(model_dir, tokenized_sent, device, args, is_roberta=False):
    if args.use_ExplainableModel:
      dataloader = DataLoader(tokenized_sent,
      batch_size=16,
      collate_fn=partial(collate_to_max_length, fill_values=[1, 0, 0])
      )
    else:
      dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)

    dirs = os.listdir(model_dir)
    dirs = sorted(dirs)

    final_output_prob, final_output_pred = [], []
    for i in range(len(dirs)):
        model_d = os.path.abspath(os.path.join(model_dir, dirs[i]))
        if args.use_bilstm_model:
            model_config = AutoConfig.from_pretrained(args.PLM)
            model_config.num_labels = 3
            model = LSTM_Model(args.PLM, model_config = model_config)
            model.load_state_dict(torch.load(os.path.join(model_d, "pytorch_model.pt")))
        elif args.use_ExplainableModel:
            model_config = AutoConfig.from_pretrained(args.PLM)
            model_config.num_labels = 3
            model = ExplainableModel(args.PLM, model_config = model_config)
            model.load_state_dict(torch.load(os.path.join(model_d, "pytorch_model.pt")))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
        
        model.eval()
        model.to(device)

        fold_prob = []
        fold_pred = []
        for data in tqdm(dataloader):
            with torch.no_grad():
                if is_roberta:
                    if args.use_ExplainableModel:
                      outputs = model(
                          input_ids=data["input_ids"].to(device),
                          start_indexs= data['start_indexs'].to(device),
                          end_indexs = data['end_indexs'].to(device),
                          span_masks = data['span_masks'].to(device)
                      )
                    else:
                      outputs = model(
                          input_ids=data["input_ids"].to(device),
                          attention_mask=data["attention_mask"].to(device),
                      )
                else:
                    outputs = model(
                        input_ids=data["input_ids"].to(device),
                        attention_mask=data["attention_mask"].to(device),
                        token_type_ids=data["token_type_ids"].to(device),
                    )
            if args.use_bilstm_model:
              logits = outputs['logits']
            else:
              logits = outputs[0]
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()

            fold_pred.extend(logits.tolist())
            fold_prob.append(prob)

        final_output_pred.append(fold_pred)
        final_output_prob.append(np.concatenate(fold_prob, axis=0).tolist())

    return final_output_pred, final_output_prob



dict_num_to_label = {0:"contradiction", 1:"entailment", 2:"neutral"}

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    for v in label:
        origin_label.append(dict_num_to_label[v])
    return origin_label


def load_test_dataset(dataset_dir, sen_preprocessor):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, train=False)
    test_dataset = preprocessing_dataset(
        test_dataset, sen_preprocessor
    )

    print(test_dataset.loc[0,'premise'])
    test_dataset['label'] = [3]*len(test_dataset["label"])

    # tokenizing dataset
    return test_dataset


def select_checkpoint(args):
    models_dir = args.model_dir
    dirs = os.listdir(models_dir)
    dirs = sorted(dirs)

    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory you want to load: ")

    checkpoint_dir = os.path.abspath(os.path.join(models_dir, dirs[int(d_idx)]))
    print("checkpoint_dir is: {}".format(checkpoint_dir))

    return checkpoint_dir


def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)

    # load my model
    model_dir = select_checkpoint(args)

    # load test datset
    test_dataset_dir = "/content/drive/MyDrive/CSV_files/open/test_data.csv"

    if args.PLM in ["klue/roberta-base", "klue/roberta-small", "klue/roberta-large", "Huffon/klue-roberta-base-nli"]:
        is_roberta = True
    else:
        is_roberta = False
    
    test_dataset = load_test_dataset(test_dataset_dir, Preprocessor)
    if args.use_ExplainableModel:
      test_dataset = ExplainableModel_Dataset(test_dataset, model_path = args.PLM)
    else:
      test_dataset = (
          NLI_Dataset(test_dataset, tokenizer, is_inference=True)
      )
      


    if args.k_fold:
        pred_answer, output_prob = inference_ensemble(
            model_dir, test_dataset, device, args, is_roberta
        )  # model에서 class 추론
        pred_answer = np.mean(pred_answer, axis=0)
        pred_answer = np.argmax(pred_answer, axis=-1)
        pred_answer = num_to_label(pred_answer)
        output_prob = np.mean(output_prob, axis=0).tolist()

    else:
        if args.use_bilstm_model:
            model_config = AutoConfig.from_pretrained(args.PLM)
            model_config.num_labels = 3
            model = LSTM_Model(args.PLM, model_config = model_config)
            model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.pt")))
        elif args.use_ExplainableModel:
            model_config = AutoConfig.from_pretrained(args.PLM)
            model_config.num_labels = 3
            model = ExplainableModel(args.PLM, model_config = model_config)
            model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.pt")))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        model.parameters
        model.to(device)

        pred_answer, output_prob = inference(
            model, test_dataset, device, args, is_roberta
        )  # model에서 class 추론
        pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # make csv file with predicted answer
   
    output = pd.DataFrame(
        {
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    sub_name = model_dir.split("/")[-1]
    output.to_csv(f"./prediction/submission_probs_{sub_name}.csv", index=False)
    output = pd.DataFrame(
        {
            "pred_label": pred_answer
        }
    )
    k1 = pd.read_csv("/content/drive/MyDrive/CSV_files/open/sample_submission.csv")
    k1['label']=output['pred_label']
    k1.to_csv(f"./prediction/submission_{sub_name}.csv", index=False)
    print("---- Finish! ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir", type=str, default="./best_models")
    parser.add_argument(
        "--PLM", type=str, help="model type (example: klue/bert-base)", required=True
    )


    parser.add_argument("--k_fold", type=int, default=0, help="not k fold(defalut: 0)")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="if want, you have to enter your model class name",
    )

    parser.add_argument(
        "--use_bilstm_model",
        type=bool,
        default=False,
        help="use bilstm_model",
    )

    parser.add_argument(
        "--preprocessing_cmb", nargs="+", help="<Required> Set flag (example: 0 1 2)"
    )

    parser.add_argument(
        "--mecab_flag",
        default=False,
        action="store_true",
        help="input text pre-processing (default: False)",
    )

    parser.add_argument(
      '--use_ExplainableModel',
      type=bool,
      default=False,
      help="use ExplainableModel",
    )


    args = parser.parse_args()
    print(args)

    os.makedirs("./prediction", exist_ok=True)
    main(args)
