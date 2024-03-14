import numpy as np
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GenerativeModel:
    def __init__(self, model_path, tokenizer_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    def generate_actions(self, state):
        input_text = f"Current state: {state}\nGenerate possible actions:\n"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        output = self.model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=5,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

        generated_actions = []
        for i in range(len(output)):
            action = self.tokenizer.decode(output[i], skip_special_tokens=True)
            action = action.split("\n")[0].strip()
            generated_actions.append(action)

        return generated_actions

    def simulate_action(self, state, action):
        input_text = f"Current state: {state}\nAction: {action}\nNext state: "
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        output = self.model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

        next_state = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return next_state.strip()