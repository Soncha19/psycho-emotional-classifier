from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import transformers
from preprocess import preprocess_text


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Завантаження моделей
with torch.serialization.safe_globals([transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2ForSequenceClassification]):
    deberta_model = torch.load('deberta_full_model.pt', map_location=device, weights_only=False)
    roberta_model = torch.load('roberta_full_model.pt', map_location=device, weights_only=False)

deberta_model.to(device).eval()
roberta_model.to(device).eval()

deberta_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Функція передбачення
def ensemble_predict(texts):
    deberta_inputs = deberta_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    roberta_inputs = roberta_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        deberta_logits = deberta_model(**deberta_inputs).logits
        roberta_logits = roberta_model(**roberta_inputs).logits
        avg_logits = (deberta_logits + roberta_logits) / 2
        probs = torch.softmax(avg_logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    return preds.cpu().numpy(), probs.cpu().numpy()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text_input']
        cleaned_text = preprocess_text(text)

        prediction, probabilities = ensemble_predict([cleaned_text])
        predicted_class = prediction[0]
        label_mapping = {0: "ADHD", 1: "Anxiety", 2: "Bipolar", 3: "BPD", 4: "Depression", 5: "Normal", 6: "OCD", 7: "PTSD"}
        predicted_label = label_mapping.get(predicted_class, "Unknown")
        predicted_probability = probabilities[0][predicted_class]

        return render_template('index.html', prediction=predicted_label, probability=f"{predicted_probability:.4f}", text_input=text)

if __name__ == "__main__":
    app.run(debug=True)
