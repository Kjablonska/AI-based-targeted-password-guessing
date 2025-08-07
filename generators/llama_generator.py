from generator import Generator
from transformers import pipeline

class LLamaGenerator(Generator):
    def __init__(self, model, tokenizer, device, guesses, max_length, temperature, top_k, top_p, remove_duplicates):
        super().__init__(model, tokenizer, device, guesses, max_length, temperature, top_k, top_p, remove_duplicates)
        self.text_generator = pipeline("text-generation", device=self.device, model=self.model, tokenizer=self.tokenizer)
        print(f"Pipeline is running on device: {self.text_generator.device}")

    def generate(self, input_text):
        output = self.text_generator(input_text,
                        return_full_text=False,
                        max_new_tokens=self.max_length,
                        do_sample=True,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        bos_token_id=self.tokenizer.encode('<s>'),
                        eos_token_id=self.tokenizer.encode('</s>'),
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=self.guesses)

        return self.process_response(output)
    

    def process_response(self, output):
        out = []
        for el in output:
            password = el["generated_text"].strip()
            out.append(password)

        out_unique = self.remove_duplicates_from_results(out)

        return out_unique