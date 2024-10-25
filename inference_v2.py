from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from typing import List, Any, Dict
import evaluate
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

dataset = load_dataset("Jinyan1/COLING_2025_MGT_en")

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
max_length=128
text_name='text'

def preprocess_function(examples):

    if isinstance(text_name, str):
        #print('vao dau')
        d = examples[text_name]
    else:
        d = examples[text_name[0]]
        for n in text_name[1:]:
            nd = examples[n]
            assert len(d) == len(nd)
            for i, t in enumerate(nd):
                d[i] += '\n' + t
    return tokenizer(d, padding='longest', max_length=max_length, truncation=True)
columns_to_remove = [col for col in dataset['dev'].column_names if col not in ['label','id']]
print("Columns to remove during preprocessing:", columns_to_remove)
tokenized_ds = dataset['dev'][:10].map(preprocess_function, batched=True, remove_columns=columns_to_remove)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# Custom collate_fn to keep 'id' along with tokenized data
class CustomDataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        # Extract 'id' from the features
        ids = [feature['id'] for feature in features]
        
        # Remove 'id' before passing the rest to the actual data collator
        features_without_id = [{k: v for k, v in f.items() if k != 'id'} for f in features]
        
        # Apply the default collator to process the tokenized fields
        batch = self.data_collator(features_without_id)
        
        # Add 'id' back to the batch
        batch['id'] = ids
        return batch

# Create the DataLoader using the custom collator
custom_data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
   '../moeAI/MGT_en_llama_7b/checkpoint-381730', num_labels=len(label2id), id2label=id2label, label2id=label2id
)
id2label = {0: "human", 1: "machine"}
label2id = {v: k for k, v in id2label.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to appropriate device
model.to(device)
model.eval()
def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results
trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
predictions = trainer.predict(tokenized_ds)

preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("bstrai/classification_report")
results = metric.compute(predictions=preds, references=predictions.label_ids)
import pandas as pd
predictions_df = pd.DataFrame({'id': test_df.index, 'label': predictions})
predictions_df.to_json('dev.jsonl', lines=True, orient='records')