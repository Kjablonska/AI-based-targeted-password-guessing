from transformers import EarlyStoppingCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers.optimization import Adafactor, AdafactorSchedule
from trl import SFTTrainer
from peft import LoraConfig
from huggingface_hub import login
from fine_tune import FineTune
from constants import HUGGING_FACE_TOKEN, BOS_TOKEN, EOS_TOKEN, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD, MAX_MODEL_INPUTS_LENGHT_LLAMA, MAX_SEQUENCE_LENGTH


class FineTuneLLama(FineTune):
    def __init__(self, model_path, model_name, training_path):
        super().__init__(model_path, model_name, training_path)

    def load_model(self):
        login(token = HUGGING_FACE_TOKEN)
        
        print(f"Load model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)
        
        self.tokenizer.add_special_tokens({"bos_token": BOS_TOKEN})
        self.tokenizer.add_special_tokens({"eos_token": EOS_TOKEN})
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=True, device_map="auto")
        self.model.config.pretraining_tp=1
        self.model.resize_token_embeddings(len(self.tokenizer))


    def preprocess_data(self, examples):
        return self.tokenizer(examples["input"], return_tensors="pt", max_length=MAX_MODEL_INPUTS_LENGHT_LLAMA, padding="max_length", truncation=True, return_attention_mask=True)

    def train(self):
        optimizer = Adafactor(
            self.model.parameters(),
            relative_step=True,
            scale_parameter=True,
            warmup_init=True,
        )
        lr_scheduler = AdafactorSchedule(optimizer)

        early_stop_callback = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE, early_stopping_threshold=EARLY_STOPPING_THRESHOLD)

        trainer = SFTTrainer(
            model=self.model,
            optimizers=(optimizer, lr_scheduler),
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            dataset_text_field="input",
            args=self.training_args,
            tokenizer=self.tokenizer,
            packing=False,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            callbacks=[early_stop_callback],
        )

        super().train()

