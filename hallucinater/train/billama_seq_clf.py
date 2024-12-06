from billm import LlamaForSequenceClassification
# -*- coding: utf-8 -*-
import sys
import argparse

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification

from pytorch_lightning import seed_everything

seed = 42
seed_everything(seed, workers=True)

import evaluate

dataset, model_size, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = 4
learning_rate = 5e-5
lora_r = 12
batch_size = 32
max_length=128

model_name_of_path ='NousResearch/Llama-2-7b-hf'
text_name='text'
accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained(model_name_of_path, use_fast=True)
ds = load_dataset("Jinyan1/COLING_2025_MGT_multingual")
def preprocess_function(examples):
    global text_name
    if isinstance(text_name, str):
        d = examples[text_name]
    else:
        d = examples[text_name[0]]
        for n in text_name[1:]:
            nd = examples[n]
            assert len(d) == len(nd)
            for i, t in enumerate(nd):
                d[i] += '\n' + t

    return tokenizer(d, padding='longest', max_length=max_length, truncation=True)

tokenized_ds = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
num_labels =2
id2label = {0: "human", 1: "machine"}
label2id = {v: k for k, v in id2label.items()}

model = LlamaForSequenceClassification.from_pretrained(
    model_name_of_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print('finished loading model')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=output_dir,
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
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print('predictions')
# Predictions on the dev set
predictions = trainer.predict(tokenized_ds["dev"])

# Save the predictions to a file
#convert_dataset_name= '_'.join(dataset.split('/'))
with open(f"biLlama_{model_size}_mgt_dev_predictions.json", "w") as f:
    json.dump(predictions.predictions.tolist(), f)
