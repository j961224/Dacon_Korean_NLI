import os
import torch
import random
import math
import sklearn
import argparse
import pickle as pickle
import pandas as pd
import numpy as np
import wandb
import importlib
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
)

from load_data import NLI_Dataset, load_data, preprocessing_dataset, ExplainableModel_Dataset
from pathlib import Path
from functools import partial

from models.transformers_with_bilstm import LSTM_Model
from models.Explainable_Model import ExplainableModel
from models.ConcatLastHiddenModel import ConcatLastFourPoolingModel

from utils.preprocessor import Preprocessor
from utils.collate_functions import collate_to_max_length
from utils.trainer import Custom_Trainer, ExplainableModel_Trainer


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
    }




def main(args):
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)

    # dynamic padding
    dynamic_padding = DataCollatorWithPadding(tokenizer=tokenizer)

    datasets = load_data(
        args.data_paths,
        args.k_fold,
        val_ratio=args.eval_ratio if args.eval_flag else 0,
        train=True
        )

    print("complete load data!!!!!!!!")

    for fold_idx, (train_dataset, test_dataset) in enumerate(datasets):
      print(fold_idx)
      # shuffle rows
      train_dataset = preprocessing_dataset(
            train_dataset, Preprocessor
        )

      train_dataset = train_dataset.sample(
          frac=1, random_state=args.seed
      ).reset_index(drop=True)
  

      if args.use_ExplainableModel:
        train_dataset = ExplainableModel_Dataset(train_dataset, model_path = args.PLM)
      else:
        train_dataset = (
          NLI_Dataset(train_dataset, tokenizer)
        )

      test_dataset = preprocessing_dataset(
          test_dataset, Preprocessor
          )

      if args.use_ExplainableModel:
          dev_dataset = ExplainableModel_Dataset(test_dataset, model_path = args.PLM)

      else:
          dev_dataset = (
              NLI_Dataset(test_dataset, tokenizer)
          )
      
      
      # wandb
      load_dotenv(dotenv_path=args.dotenv_path)
      WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
      wandb.login(key=WANDB_AUTH_KEY)

      wandb.init(
          entity="jj961224",
          project="dacon_korea_sentence_relation",
          name=args.wandb_unique_tag + "_" + str(fold_idx),
          group=args.PLM + "-k_fold" if args.k_fold > 0 else args.PLM,
      )
      wandb.config.update(args)

      train_model(
          args,
          train_dataset,
          dev_dataset,
          fold_idx=fold_idx,
          dynamic_padding=dynamic_padding,
          tokenizer=tokenizer
      )

      wandb.finish()


def train_model(
    args,
    train_dataset,
    dev_dataset,
    fold_idx,
    dynamic_padding,
    tokenizer,
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model_config = AutoConfig.from_pretrained(args.PLM)
    model_config.num_labels = 3

    def model_init():

      if args.mlm_training:
        checkpoint = args.checkpoint
        PLM = checkpoint
      else:
        PLM = args.PLM

      if args.use_bilstm_model:
        model = LSTM_Model(PLM, model_config = model_config).to(device)
      
      elif args.use_ExplainableModel:
        model = ExplainableModel(bert_dir = PLM, model_config = model_config).to(device)
      
      elif args.use_FourHidden:
        model_config.update({"output_hidden_states": True})
        model = ConcatLastFourPoolingModel(PLM, model_config).to(device)

      else:
        model = AutoModelForSequenceClassification.from_pretrained(
            PLM,
            ignore_mismatched_sizes=args.ignore_mismatched,
            config=model_config,
        ).to(device)
      return model

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=3,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy=args.evaluation_strategy if args.eval_flag else "no",
        metric_for_best_model='accuracy',
        eval_steps=500 if args.eval_flag else 0,  # evaluation step.
        load_best_model_at_end=True if args.eval_flag else False,
        report_to="wandb",
        label_smoothing_factor=0.2,
        gradient_accumulation_steps = args.gradient_accumulation_steps
    )
    training_args.is_noam = args.is_noam
    training_args.d_model = model_config.hidden_size
    
    if args.use_ExplainableModel:
      trainer = ExplainableModel_Trainer(
          # the instantiated 🤗 Transformers model to be trained
          model_init=model_init,
          args=training_args,  # training arguments, defined above
          train_dataset=train_dataset,  # training dataset
          eval_dataset=dev_dataset if args.eval_flag else None,  # evaluation dataset
          compute_metrics=compute_metrics,  # define metrics function
          data_collator=partial(collate_to_max_length, fill_values=[1, 0, 0]),
          tokenizer=tokenizer,
      )
    else:
      trainer = Custom_Trainer(
          model_init=model_init,
          args=training_args,  # training arguments, defined above
          train_dataset=train_dataset,  # training dataset
          eval_dataset=dev_dataset if args.eval_flag else None,  # evaluation dataset
          compute_metrics=compute_metrics,  # define metrics function
          data_collator=dynamic_padding,
          tokenizer=tokenizer,
      )

    # train model
    train_result = trainer.train()

    trainer.args.output_dir = args.save_dir + args.wandb_unique_tag +"_"+str(fold_idx)
                
    trainer.save_model()
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--save_dir",
        default="./best_models",
        help="model save at save_dir/PLM-wandb_unique_tag",
    )
    parser.add_argument(
        "--PLM",
        type=str,
        default="klue/roberta-large",
        help="model type (default: klue/bert-large)",
    )
    
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of epochs to train (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="train batch size (default: 16)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="gradient_accumulation_steps (default: 1)"
    )


    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="number of warmup steps for learning rate scheduler (default: 500)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="strength of weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="evaluation strategy to adopt during training, steps or epoch (default: steps)",
    )
    parser.add_argument(
        "--ignore_mismatched",
        type=bool,
        default=False,
        help="ignore mismatched size when load pretrained model",
    )

    # Validation
    parser.add_argument(
        "--eval_flag",
        default=True,
        action="store_true",
        help="eval flag (default: False)",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.2,
        help="eval data size ratio (default: 0.2)",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="eval batch size (default: 16)"
    )

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

    # Wandb
    parser.add_argument(
        "--dotenv_path", default="/opt/ml/wandb.env", help="input your dotenv path"
    )
    parser.add_argument(
        "--wandb_unique_tag",
        default="klue-roberta-large-batch64-epoch5-lr2e5-ls0.2_kfold",
        help="input your wandb unique tag (default: roberta-large-default)",
    )

    parser.add_argument("--k_fold", type=int, default=0, help="not k fold(defalut: 0)")
    parser.add_argument(
        "--is_noam",
        type=bool,
        default=False,
        help="use noam scheduler",
    )

    parser.add_argument(
        "--use_bilstm_model",
        type=bool,
        default=False,
        help="use bilstm_model",
    )

    parser.add_argument(
      '--use_ExplainableModel',
      type=bool,
      default=False,
      help="use ExplainableModel",
    )

    parser.add_argument(
      '--use_FourHidden',
      type=bool,
      default=False,
      help="use FourHiddenModel",
    )

    parser.add_argument(
      '--ConcatLastFourCLSModel',
      type=bool,
      default=False,
      help="use ConcatLastFourCLSModel",
    )

    parser.add_argument(
      '--data_paths',
      type=str,
      default=None,
      help="data paths",
    )

    parser.add_argument(
      '--mlm_training',
      type=bool,
      default=False,
      help="whether mlm training",
    )


    args = parser.parse_args()

    # Start
    seed_everything(args.seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)
