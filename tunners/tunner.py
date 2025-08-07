from constants import LEARNING_RATE, PER_DEVICE_EVAL_BATCH_SIZE, PER_DEVICE_TRAIN_BATCH_SIZE, NUM_TRAIN_EPOCH, EVAL_STEPS, LOGGING_STEPS, SAVE_TOTAL_LIMIT, WARMUP_RATIO, WEIGHT_DECAY
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments
from callback import PlotCallback
import torch
import json

class Tunner():
    def __init__(self, model_path, model_name, training_path):
        self.model_path = model_path
        self.dataset = None
        self.model = None
        self.model_name = model_name
        self.training_path = training_path
        self.trainer = None
        self.tokenizer = None
        self.tokenized_train_dataset = None
        self.tokenized_test_dataset = None
        self.callback = PlotCallback(self.training_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()

    def load_data(self):
        print(f"Load dataset from {self.training_path}...")
        data = load_dataset("json", data_files={"data": self.training_path})

        print("Shuffling set...")
        data.shuffle()

        self.dataset = data["data"].train_test_split(test_size=0.1)


    def preprocess_data(self, data):
        model_inputs = self.tokenizer(data["input"], max_length=128, padding="max_length", truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(data["output"], max_length=16, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs


    def evaluate(self):
        eval_results = self.trainer.evaluate()
        logs = self.trainer.state.log_history

        with open(f'{self.model_path}/trainer-eval.json', 'w') as f:
            json.dump([eval_results], f)

        with open(f'{self.model_path}/trainer-logs.json', 'w') as f:
            json.dump(logs, f)

    def create_trainer(self):
        print("Creating training arguments")
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=f"{self.model_path}/training",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            num_train_epochs=NUM_TRAIN_EPOCH,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            logging_dir=f"{self.model_path}/logs",
            logging_steps=LOGGING_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY
        )

    def train(self):
        print("Start training...")
        self.trainer.train()
        print("Training completed. Saving model...")
        self.trainer.save_model(f"{self.model_path}")
        self.tokenizer.save_pretrained(f"{self.model_path}")
        print("End")
