from   datasets import load_dataset
from   transformers import TrainingArguments, AutoModelForSequenceClassification
from   unsloth import FastLanguageModel
from   peft import LoraConfig, get_peft_model
import torch

# Prepare dataset
dataset = load_dataset('ag_news')

# Load and quantize the Llama 3.2 Model
model_name = "unsloth/llama-3-8b-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=True
)

# Add a Classification Head
num_labels = 4  # Number of classes in your dataset
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    torch_dtype=torch.bfloat16
)

# Configure LoRA for Fine-Tuning
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# Pre-process the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Initialize the trainer and train
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer
)

trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
