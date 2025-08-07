import Levenshtein
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import json
import torch
import time

from generators.generator import Generator
from generators.llama_generator import LLamaGenerator

class Results():
    def __init__(self, test_set, model_path, out_path, model_name, guesses, temperature, top_k, top_p, remove_duplicates = True):
        self.remove_duplicates = remove_duplicates
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.out_path = out_path
        self.test_set = load_dataset("json", data_files={test_set})["test"]

        self.model_path = model_path
        self.model_name = model_name
        self.guesses = guesses
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_model_tokenizer()

        self.guessed_passwords = {}
        self.results = {}
        self.analysis_output = {}

        # If real password is provided, guesses analysis can be performed.
        self.can_analyse = False
        

    def init_model_tokenizer(self):
        if "Llama" in self.model_name:
            print("Generator for Llama")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.generator = LLamaGenerator(self.model, self.tokenizer, self.device, self.guesses, 5, self.temperature, self.top_k, self.top_p, self.remove_duplicates)
        elif "T5" in self.model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path, local_files_only=True)
            self.model.to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path, local_files_only=True)
            print(f"Generator for {self.model_name}")
            self.generator = Generator(self.model, self.tokenizer, self.device,  self.guesses, 16, self.temperature, self.top_k, self.top_p, self.remove_duplicates)
        elif "BART" in self.model_name:
            self.tokenizer = BartTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.generator = Generator(self.model, self.tokenizer, self.device,  self.guesses, 16, self.temperature, self.top_k, self.top_p, self.remove_duplicates)
    
    def write_only_guesses(self, path):
        with open(path, "w") as f:
            for _, value in self.results.items():
                for el in value["guesses"]:
                    f.write(el)
                    f.write("\n")


    def get_results(self):
        sample = self.test_set
        print(f"Number of samples: {len(sample)}")
        inputs = sample["input"]

        # If password is present as output in the test set, it can be later used to analyse the generated guesses.
        if "output" in sample:
            outputs = sample["output"]
            self.can_analyse = True
        else:
            outputs = "N/A"

        emails = sample["email"]

        i = 0
        start_time = time.time()
        while i < len(inputs):
            out = self.generator.generate(inputs[i])
            self.results[inputs[i]] = {"email": emails[i], "password": outputs[i], "guesses": out}
            i += 1

        end_time = time.time()
        self.time = end_time - start_time
        print(f"Generation complete:\nModel: {self.model_name}\nSamples: {len(inputs)}\nNumber of guesses: {self.guesses}\nTime: {self.time}")

        with open(f'{self.out_path}/all-guesses.jsonl', 'w') as fp:
            for key, value in self.results.items():
                out = {"prompt": key, "results_count": len(value["guesses"]),  "results": value}
                json.dump(out, fp)
                fp.write('\n')
    
    def get_best_guess_metrics(self, best_guess_score, best_guess, best_iteration, current_guess_score, current_guess, current_iteration):
        if current_guess_score > best_guess_score:
            return current_guess_score, current_guess, current_iteration
        else:
            return best_guess_score, best_guess, best_iteration

    def calculate_metrics(self):
        if not self.can_analyse:
            print("Can not perform analysis. Please verify if provided input data is complete and results are not empty.")
            return
        
        guessed_passwords = 0
        for key, value in self.results.items():
            real_password = value["password"]
            email = value["email"]

            max_score = 0
            levens_guesses = 0
            guessed_pass = ''


            guesses = value["guesses"]
            for j, password in enumerate(guesses):
                current_score = Levenshtein.ratio(real_password, password)
                max_score, guessed_pass, levens_guesses = self.get_best_guess_metrics(max_score, guessed_pass, levens_guesses, current_score, password, j+1)

                if password == real_password:
                    self.guessed_passwords[key] = {"email": email, "password": password, "guess_nb": j + 1}
                    break
            

            self.analysis_output[key] = {"email": email, "real_password": real_password, "guessed_passwords": guessed_passwords, "scores": {"max_score": max_score, "best_pass": guessed_pass, "guess_number": levens_guesses}}             

            with open(f'{self.out_path}/eval.jsonl', 'w') as fp:
                for key, value in self.analysis_output.items():
                    out = {"prompt": key, "metrics": value}
                    json.dump(out, fp)
                    fp.write('\n')
                
            with open(f'{self.out_path}/guessed.jsonl', 'w') as fp:
                for key, value in self.guessed_passwords.items():
                    username = value["email"].split("@")[0]

                    out = {"email": key, "metrics": value, "username_similarity_score": Levenshtein.ratio(username, value["password"])}
                    json.dump(out, fp)
                    fp.write('\n')


def geneate_results(data):
    results = Results(data["test_set_path"], data['model_path'], data["out_path"], data["name"], data["guesses"], data["temperature"], data["top_k"], data["top_p"], False)
    
    with open(f"{data['out_path']}/config.jsonl", "w") as cnf:
        json.dump(data, cnf)

    print("Generating guesses...")
    results.get_results()

    print("Analysing guesses...")
    results.calculate_metrics()


if __name__== "__main__":
    print(time.time())
    if len(sys.argv) < 3:
        raise Exception("Provide arguments")
    
    with open(sys.argv[1], "r") as file:
        data = json.load(file)

    print("Generate results for:")
    print(data[1]["test"][int(sys.argv[2])])
    data = data[1]["test"][int(sys.argv[2])]
    geneate_results(data)
