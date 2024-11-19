import os
import numpy as np
import pandas as pd
import torch
from   trl import SFTTrainer
from   transformers import TrainingArguments, TextStreamer
from   unsloth.chat_templates import get_chat_template
from   unsloth import FastLanguageModel
from   datasets import Dataset
from   unsloth import is_bfloat16_supported
from   accelerate import Accelerator
from   datetime import datetime

# Saving model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Warnings
import warnings
warnings.filterwarnings("ignore")

max_seq_length = 1024
accelerator = Accelerator(cpu=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.float16, # Ensure weights are in FP16
    device_map="auto",  # Automatically place layers on GPU/CPU
    offload_folder="./offload",  # Directory for disk offloading (optional)
    offload_state_dict=True,  # Save state dict to disk
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)
model.gradient_checkpointing_enable()
print(model.print_trainable_parameters())

data = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)
data['Context_length'] = data['Context'].apply(len)
filtered_data = data[data['Context_length'] <= 1500]

# Add context and response lengths as new columns in the DataFrame
data['ln_Context'] = data['Context'].apply(len)
data['ln_Response'] = data['Response'].apply(len)

# Filter the data based on combined sequence length
filtered_data = data[(data['ln_Context'] + data['ln_Response']) <= max_seq_length]
ln_Response = filtered_data['Response'].apply(len)

data_prompt = """Analyze the provided text from a mental health perspective. Identify any indicators of emotional distress, coping mechanisms, or psychological well-being. Highlight any potential concerns or positive aspects related to mental health, and provide a brief explanation for each observation.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"  # Fallback to default if eos_token is None
def formatting_prompt(examples):
    inputs = examples["Context"]
    outputs = examples["Response"]
    texts = [
        data_prompt.format(input_, output) + EOS_TOKEN
        for input_, output in zip(inputs, outputs)
    ]
    return {"text": texts}

training_data = Dataset.from_pandas(filtered_data)
training_data = training_data.map(formatting_prompt, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args = TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=40,
        fp16=True,  # Use FP16 mixed precision
        optim="adamw_8bit",  # Use memory-efficient optimizer
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        logging_steps=1,
        save_total_limit=2,
        report_to="none",
    ),
)

trainer.train()

# Save the trained model
savefile = f"lora_{datetime.now().strftime('%Y%m%d%H%M%S')}"
model.save_pretrained(savefile)

text="I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone?"

model = FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    data_prompt.format(
        #instructions
        text,
        #answer
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 5020, use_cache = True)
answer=tokenizer.batch_decode(outputs)
answer = answer[0].split("### Response:")[-1]
print("Answer of the question is:", answer)
