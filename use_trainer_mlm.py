from transformers import BertConfig, BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import transformers
transformers.set_seed(1337)
from datasets import load_dataset
from tqdm.auto import tqdm
import re
from pathlib import Path
import training


dataset_name = "wikipedia"
dataset_version = "20200501.en"
dataset = load_dataset(dataset_name, dataset_version)

model = BertForMaskedLM(BertConfig())
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True
)

args = TrainingArguments(
    output_dir="./trainer_results",
    overwrite_output_dir=False,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    save_steps=100,
    save_total_limit=3,
    seed=1337,
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./trainer_results")