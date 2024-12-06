from .model_llama_for_sequence_classification import llama_for_sequence_clasification
from peft import LoraConfig, TaskType

def setup(config):
    """
    Set up model configurations like LoRA.
    """
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=config['lora_r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'])
    
    # Additional model setup or configurations can be added here if necessary.
    
    return peft_config

def load_model(config):
    """
    This function loads the model and applies necessary configurations.
    """
    if config['model_name'] == "llama":
        # Set up the model config (e.g., LoRA)
        peft_config = setup(config)
        
        # Load the model using the specific model loader
        model, tokenizer = llama_for_sequence_clasification(config, peft_config)
        
        return model, tokenizer
    else:
        raise ValueError(f"Model {model_name} is not supported yet.")
