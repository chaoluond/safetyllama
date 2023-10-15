import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, TrainerCallback, default_data_collator, Trainer, TrainingArguments
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.configs.datasets import safetyllama_finetune_dataset
from contextlib import nullcontext

ENABLE_PROFILER = False
OUTPUT_DIR = "tmp/Llama-2-7b-chat-safety"
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# Step 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, torch_dtype = "auto", device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Step 2. Load training dataset
train_dataset = get_preprocessed_dataset(tokenizer, safetyllama_finetune_dataset, 'train')

# Step 3. PEFT configuration
model.train()
model, lora_config = create_peft_config(model)

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 1,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': 2,
    'gradient_checkpointing': False,
}

# Step 4. [Optional] Define a profiler
if ENABLE_PROFILER:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{OUTPUT_DIR}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    
    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler
            
        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()
    
# Step 5. Finetune model
# Define training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    bf16=True,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused",
    max_steps=total_steps if ENABLE_PROFILER else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[profiler_callback] if ENABLE_PROFILER else [],
    )

    # Start training
    trainer.train()
    # Push finetuned model checkpoint to huggingface
    trainer.push_to_hub()
    
# Save and push model checkpoint
model.save_pretrained(OUTPUT_DIR)






















