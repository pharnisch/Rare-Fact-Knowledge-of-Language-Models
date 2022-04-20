from transformers import BertModel, BertConfig, BertTokenizer, BertForPreTraining
from transformers import DataCollatorForLanguageModeling
from transformers.data.datasets.language_modeling import TextDatasetForNextSentencePrediction
from transformers import Trainer, TrainingArguments


model = BertForPreTraining(BertConfig())
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True
)

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="./training/data/nsp/wikipedia/20200501.en/text_for_nsp.txt_0",
    block_size=512,  # TODO: what does this, what number is suitable for me?
    overwrite_cache=False,
    short_seq_probability=0.1,
    nsp_probability=0.5
)

args = TrainingArguments(
    output_dir="./trainer_results",
    overwrite_output_dir=False,
    num_train_epochs=10,  # TODO: how many? although it is limitation anyway...
    per_device_train_batch_size=32,  # TODO: what does this?
    save_steps=100,
    save_total_limit=3,
    seed=1337,
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer
)
trainer.train()
trainer.save_model("./trainer_results")