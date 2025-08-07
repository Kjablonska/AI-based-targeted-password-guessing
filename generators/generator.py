from collections import OrderedDict

class Generator():
    def __init__(self, model, tokenizer, device, guesses, max_length, temperature, top_k, top_p, remove_duplicates=False):
        print("Init generator")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.guesses = guesses
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.remove_duplicates = remove_duplicates

        if remove_duplicates == True:
            self.generate_generic()
    
    def decode(self, output_ids):
        out = []
        for output_id in output_ids:
            output_text = self.tokenizer.decode(output_id, skip_special_tokens=True)
            out.append(output_text)

        if self.remove_duplicates:
            return self.remove_duplicates_from_results(out)
        
        return out

    def remove_duplicates_from_results(self, arr):
        return list(OrderedDict.fromkeys(arr))

    def model_generate(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device).input_ids
        return self.model.generate(input_ids, max_length=self.max_length, 
                do_sample=True,
                num_beams=1,
                num_return_sequences=self.guesses, 
                                    temperature=self.temperature,
                                    top_k=self.top_k,
                                    top_p=self.top_p)
    
    def generate(self, input_text):
        output_ids = self.model_generate(input_text)
        out = self.decode(output_ids)

        if len(out) < self.guesses:
            to_generate_nb = self.guesses - len(out)
            out += self.generic_passwords[:to_generate_nb]
        
        out_unique = self.remove_duplicates_from_results(out)

        return out_unique