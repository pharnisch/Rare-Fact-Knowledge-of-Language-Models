from transformers import BertConfig, BertTokenizer, BertForPreTraining
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import transformers
transformers.set_seed(1337)

#from transformers.data.datasets.language_modeling import TextDatasetForNextSentencePrediction
from training.data.language_modeling import TextDatasetForNextSentencePrediction

model = BertForPreTraining(BertConfig())
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="./training/data/wikipedia/20200501.en/nsp/text_for_nsp.txt",
    block_size=512,
    overwrite_cache=False,
    short_seq_probability=0.1,
    nsp_probability=0.5
)

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