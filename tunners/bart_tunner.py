from transformers import EarlyStoppingCallback, Trainer,BartTokenizer, BartForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule
from tunner import Tunner
from constants import EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD, MAX_LABELS_LENGTH, MAX_MODEL_INPUTS_LENGHT


class BARTTunner(Tunner):
    def __init__(self, model_path,model_name, training_path):
        super().__init__(model_path, model_name, training_path)

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name, device_map="auto", ignore_mismatched_sizes=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_data(self):
        super().load_data()
        self.tokenized_train_dataset = self.dataset["train"].map(self.preprocess_data, batched=True)
        self.tokenized_test_dataset = self.dataset["test"].map(self.preprocess_data, batched=True)

    def preprocess_data(self, examples):
        model_inputs = self.tokenizer(examples["input"], max_length=MAX_MODEL_INPUTS_LENGHT, padding="max_length", truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["output"], max_length=MAX_LABELS_LENGTH, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self):        
        optimizer = Adafactor(
            self.model.parameters(),
            relative_step=True,
            scale_parameter=True,
            warmup_init=True,
        )

        lr_scheduler = AdafactorSchedule(optimizer)
        early_stop_callback = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE, early_stopping_threshold=EARLY_STOPPING_THRESHOLD)
        
        self.trainer = Trainer(
            model=self.model,
            optimizers=(optimizer,lr_scheduler),
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            callbacks=[early_stop_callback],
        )

        super().train()
        