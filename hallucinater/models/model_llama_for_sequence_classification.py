from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification
from peft import get_peft_model

def llama_for_sequence_clasification(config, peft_config):
    """
    Load the Llama model with the given configuration and apply LoRA.
    """
    # Load the tokenizer
    model_id = get_pretrain_model(config['model_size'])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    id2label = config['id2label']
    label2id = {v: k for k, v in id2label.items()}
    
    # Load the model for sequence classification
    model = LlamaForSequenceClassification.from_pretrained(
        model_id, num_labels=len(id2label), id2label=id2label, label2id=label2id
    ).bfloat16()  # Adjust precision here based on your needs

    # Apply LoRA (Low-Rank Adaptation) configuration to the model
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    print('Finished loading model')
    return model, tokenizer
    
def get_pretrain_model(model_size):
    if model_size.lower() == '7b':
        model_id = 'NousResearch/Llama-2-7b-hf'
    elif model_size.lower() == '13b':
        model_id = 'NousResearch/Llama-2-13b-hf'
    return model_id