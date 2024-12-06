from datasets import load_dataset
import os


def load_task(config):
    dataset = load_dataset(config['dataset'])
    return dataset

def load_dataset_from_json_file(data_files):
    # Extract the file extension
    file_extension = os.path.splitext(data_files)[1][1:]  # Removes the dot
    dataset = load_dataset('json', data_files=data_files)
    return dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(examples, text_name, tokenizer, max_length):
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