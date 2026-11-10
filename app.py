
import download_models

from flask import Flask, render_template, request
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# ==========================
# DEVICE
# ==========================

device = torch.device("cpu")

# ==========================
# LOAD SVM MODEL
# ==========================

vectorizer = joblib.load("tfidf_vectorizer.pkl")
svm_model = joblib.load("svm_model.pkl")

print("SVM dan TF-IDF berhasil dimuat.")

# ==========================
# LOAD BERT MODEL
# ==========================

model_name = "bert-base-multilingual-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

state_dict = torch.load(
    "bert_hoax_classifier.pth",
    map_location=device
)

bert_model.load_state_dict(state_dict, strict=True)

bert_model.to(device)
bert_model.eval()

print("BERT hasil training berhasil dimuat.")

# ==========================
# PREDICT FUNCTION (SVM)
# ==========================

def predict_svm(text):

    vec = vectorizer.transform([text])

    pred = svm_model.predict(vec)[0]

    return "Hoax" if pred == 0 else "Valid"


# ==========================
# PREDICT FUNCTION (BERT)
# ==========================

def predict_bert(text):

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = bert_model(**inputs)

        pred = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    return "Hoax" if pred == 0 else "Valid"


# ==========================
# ROUTE
# ==========================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    model_choice = "bert"
    news_text = ""
    error_message = None

    if request.method == "POST":

        news_text = request.form.get("news", "").strip()
        model_choice = request.form.get("model", "bert")

        if model_choice not in ["bert", "svm"]:
            model_choice = "bert"
            error_message = "Pilihan model tidak valid."

        elif not news_text:
            error_message = "Silakan masukkan teks berita terlebih dahulu."

        else:

            if model_choice == "svm":
                prediction = predict_svm(news_text)

            else:
                prediction = predict_bert(news_text)

    return render_template(
        "index.html",
        prediction=prediction,
        model_choice=model_choice,
        news_text=news_text,
        error_message=error_message
    )


# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
