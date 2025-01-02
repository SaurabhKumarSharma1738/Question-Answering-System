from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


class QuestionAnswerApp(App):
    def build(self):
        # Root layout for the Kivy app
        self.layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Context input
        self.context_label = Label(text="Enter the context:")
        self.context_input = TextInput(hint_text="Provide the context here", multiline=True)

        # Question input
        self.question_label = Label(text="Enter your question:")
        self.question_input = TextInput(hint_text="Type your question here", multiline=False)

        # Output label
        self.answer_label = Label(text="Answer will appear here")

        # Button to generate answer
        self.submit_button = Button(text="Get Answer", on_press=self.get_answer)

        # Add widgets to the layout
        self.layout.add_widget(self.context_label)
        self.layout.add_widget(self.context_input)
        self.layout.add_widget(self.question_label)
        self.layout.add_widget(self.question_input)
        self.layout.add_widget(self.submit_button)
        self.layout.add_widget(self.answer_label)

        # Load the Hugging Face model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

        return self.layout

    def get_answer(self, instance):
        """Generate an answer using the Hugging Face model."""
        question = self.question_input.text.strip()
        context = self.context_input.text.strip()

        if not question or not context:
            self.answer_label.text = "Please provide both a question and context."
            return

        # Tokenize input
        inputs = self.tokenizer(question, context, return_tensors="pt")

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1

        # Decode the answer
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        )

        # Display the answer
        self.answer_label.text = f"Answer: {answer}"


if __name__ == "__main__":
    QuestionAnswerApp().run()
