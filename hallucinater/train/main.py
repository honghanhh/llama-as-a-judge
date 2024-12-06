# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 

import evaluate
import json
import numpy as np
import torch
from utils import load_task, preprocess_function, compute_metrics
from hallucinater.models import load_model
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

from pytorch_lightning import seed_everything


if __name__ == "__main__":
    seed = 42
    seed_everything(seed, workers=True)
    
    # Load configuration file 
    with open('hallucinater/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    output_dir = config["outdir"]
    text_name = "text"
    train_name = "train"
    dev_name = "dev"

    # Load the model and tokenizer based on the model name
    model, tokenizer = load_model(config)
    model.print_trainable_parameters()

    # Load dataset
    ds = load_task(config)
    tokenized_ds = ds.map(preprocess_function, batched=True,fn_kwargs={"text_name": text_name, "tokenizer": tokenizer, "max_length": config["max_length"]})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print('Start training ...')
    
    training_args = TrainingArguments(
        output_dir= output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds[train_name],
        eval_dataset=tokenized_ds[dev_name],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    # Predictions on the dev set
    predictions = trainer.predict(tokenized_ds[dev_name])
    
    # Save the predictions to a file
    with open("predictions.json", "w") as f:
        json.dump(predictions.predictions.tolist(), f)

