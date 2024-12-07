import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 
import yaml
import numpy as np
import pandas as pd
from hallucinater.train.utils import load_task, preprocess_function, compute_metrics, load_dataset_from_json_file
from hallucinater.models import load_model
from transformers import Trainer, DataCollatorWithPadding
from pytorch_lightning import seed_everything


def main():
    # Load configuration file 
    with open('hallucinater/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    text_name = "text"
    test_name = "train"
    # Load the model and tokenizer based on the model name
    model, tokenizer = load_model(config)    
    model.eval()
    ds = load_dataset_from_json_file(config["test_file"])
    tokenized_ds = ds.map(preprocess_function, batched=True,fn_kwargs={"text_name": text_name, "tokenizer": tokenizer, "max_length": config["max_length"]})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_ds['train'])
    preds = np.argmax(predictions.predictions, axis=-1)
    predictions_df = pd.DataFrame({'id': tokenized_ds[test_name]['testset_id'], 'label': preds})
    predictions_df.to_json('output.jsonl', lines=True, orient='records')
if __name__ == "__main__":
    main()
