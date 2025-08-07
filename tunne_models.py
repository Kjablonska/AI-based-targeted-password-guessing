import json
import sys
from fine_tune_t5 import FineTuneT5
from fine_tune_llama import FineTuneLLama
from fine_tune_bart import FineTuneBart
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

class Main():
    def __init__(self, tunner):
        self.tunner = tunner

    def fine_tune(self):
        self.tunner.load_model()
        self.tunner.load_data()
        self.tunner.create_trainer()
        self.tunner.train()

if __name__== "__main__":
    if len(sys.argv) < 3:
        raise Exception("Provide arguments")
    
    # Open training config file
    with open(sys.argv[1], "r") as file:
        data = json.load(file)
    
    data = data[0]["train"][int(sys.argv[2])]
    with open(f"{data['model_path']}/config.jsonl", "w") as cnf:
        json.dump(data, cnf)

    if data["name"] == "BART":
        tunner = FineTuneBart(model_path=data["model_path"], model_name=data["model_name"], training_path=data["training_set_path"])
    if data["name"] == "T5":
        tunner = FineTuneT5(model_path=data["model_path"], model_name=data["model_name"], training_path=data["training_set_path"])
    if data["name"] == "Llama":
        tunner = FineTuneLLama(model_path=data["model_path"], model_name=data["model_name"], training_path=data["training_set_path"])

    main = Main(tunner)
    main.fine_tune()

