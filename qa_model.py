from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

class QuestionAnsweringModel:
    def __init__(self):
        # Load pre-trained DistilBERT model and tokenizer
        self.model_name = "distilbert-base-uncased-distilled-squad"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_name)

    def answer_question(self, context, question):
        # Tokenize input
        inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]

        # Perform inference
        outputs = self.model(**inputs)
        start_scores, end_scores = outputs.start_logits, outputs.end_logits

        # Get the most likely answer tokens
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1

        # Convert tokens to string
        answer_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx])
        return self.tokenizer.convert_tokens_to_string(answer_tokens)
