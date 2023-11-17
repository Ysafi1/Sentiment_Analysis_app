from transformers import RobertaTokenizer, RobertaForSequenceClassification
import gradio as gr
import torch

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("KABANDA18/FineTuning-Roberta-base_Model")
model = RobertaForSequenceClassification.from_pretrained("KABANDA18/FineTuning-Roberta-base_Model")

def sentiment_analysis(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass through the model
    with torch.no_grad():
        output = model(**inputs)

    # Extract the predicted probabilities
    scores = torch.nn.functional.softmax(output.logits, dim=1).squeeze().tolist()

    # Define the sentiment labels
    labels = ["Negative", "Neutral", "Positive"]

    # Create a dictionary of sentiment scores
    scores_dict = {label: score for label, score in zip(labels, scores)}

    return scores_dict

demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Write/Type your tweet here"),
    outputs="text",
    #intrepretation="default",
    examples=[
        ["Covid Vaccine are Health"],
        ["There's a global pandemic ongoing called Covid"],
        ["Covid is dangerous"],
        ["Covid is affecting Businesses badly"],
        ["This so-called Covid is not going to block our shine. Come to The beach this weekend! It's going to be lit"],
    ],
    title="Covid Tweets Sentiment Analysis App",
    description="This Application is the interface to Our Sentiment Analysis Model fine-tuned from a Roberta-base model.",
)

demo.launch()
