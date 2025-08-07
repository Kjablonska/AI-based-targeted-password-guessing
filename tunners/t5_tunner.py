from transformers import Seq2SeqTrainer, EarlyStoppingCallback, T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule
from fine_tune import FineTune
from constants import EARLY_STOPPING_THRESHOLD, EARLY_STOPPING_PATIENCE

class FineTuneT5(FineTune):
    def __init__(self, model_path, model_name, training_path):
        print(model_path, model_name, training_path)
        super().__init__(model_path, model_name, training_path)

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, char_level=True, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

        self.model.resize_token_embeddings(len(self.tokenizer))
    

    def load_data(self):
        super().load_data()
        self.tokenized_train_dataset = self.dataset["train"].map(self.preprocess_data, batched=True)
        self.tokenized_test_dataset = self.dataset["test"].map(self.preprocess_data, batched=True)

    def train(self):
        optimizer = Adafactor(
            self.model.parameters(),
            relative_step=True,
            scale_parameter=True,
            warmup_init=True,
        )
        lr_scheduler = AdafactorSchedule(optimizer)
        
        early_stop_callback = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE, early_stopping_threshold=EARLY_STOPPING_THRESHOLD)
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            optimizers=(optimizer,lr_scheduler),
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            callbacks=[early_stop_callback],
        )        

        super().train()
        