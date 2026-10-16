try:
    import spaces
except ImportError:
    spaces = None

import re

import gradio as gr
import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HUB_ID = "DavidPontientKayumba/afroxlmr-kinnews-topic-classifier"
MAX_LEN = 256

LABEL_NAMES = ["politics", "sport", "economy", "health", "entertainment", "history",
               "technology", "tourism", "culture", "fashion", "religion",
               "environment", "education", "relationship"]

# The same cleaning that was used for training in notebook 1
URL = re.compile(r"https?://\S+|www\.\S+")
NOT_WORD = re.compile(r"[^\w\s']")
SPACES = re.compile(r"\s+")


def clean_text(text):
    text = text.lower()
    text = URL.sub(" ", text)
    text = NOT_WORD.sub(" ", text)
    return SPACES.sub(" ", text).strip()


# Model 1: the baseline (TF-IDF character n-grams and a linear SVM)
baseline = joblib.load("best_baseline.joblib")
vectorizer = baseline["vectorizer"]
svm = baseline["classifier"]

# Model 2: AfroXLMR fine tuned on KINNEWS, loaded from the Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(HUB_ID)
model = AutoModelForSequenceClassification.from_pretrained(HUB_ID)
model.eval()


if spaces is not None:
    # Free Hugging Face accounts can only host Gradio apps on ZeroGPU hardware, and ZeroGPU
    # expects at least one function marked for the GPU. This function does nothing and is never
    # called. Both models run on the CPU, so no GPU time is used.
    @spaces.GPU
    def gpu_placeholder():
        return "ok"


def classify(text):
    if not text or len(text.split()) < 3:
        return {}, "Please enter at least a few words of Kinyarwanda news text.", ""

    cleaned = clean_text(text)

    # Fine tuned model: real probabilities from the softmax layer
    n_tokens = len(tokenizer(cleaned)["input_ids"])
    enc = tokenizer(cleaned, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    fine_tuned = {LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)}
    fine_tuned_top = max(fine_tuned, key=fine_tuned.get)

    # Baseline: the SVM gives margins, which are not probabilities
    margins = svm.decision_function(vectorizer.transform([cleaned]))[0]
    order = np.argsort(margins)[::-1][:3]
    baseline_top = LABEL_NAMES[int(svm.classes_[order[0]])]
    baseline_text = "Predicted topic: " + baseline_top + "\n\nTop 3 by score (not probabilities):\n"
    baseline_text += "\n".join(
        str(rank + 1) + ". " + LABEL_NAMES[int(svm.classes_[i])] + "  (score " + str(round(float(margins[i]), 2)) + ")"
        for rank, i in enumerate(order))

    if baseline_top == fine_tuned_top:
        comparison = "Both models agree: " + baseline_top + "."
    else:
        comparison = ("The models disagree. Baseline: " + baseline_top +
                      ". Fine tuned model: " + fine_tuned_top + ".")
    if n_tokens > MAX_LEN:
        comparison += ("\n\nNote: this text has " + str(n_tokens) + " tokens. The fine tuned model reads only the first "
                       + str(MAX_LEN) + ", the baseline reads all of it.")

    return fine_tuned, baseline_text, comparison


EXAMPLES = [
    ["ishema ry abongereza ruhando rw i burayi bambariye guhigika mukeba manchester united ku nshuro mbere mateka liverpool manchester united abake b ibihe byose bahuriye mukino uefa europa league umukino ubanza liverpool yats"],
    ["ese koko piramide mbere ibarizwa egypte yubatswe"],
    ["umuzungu yagaragaje ubuhanga bisakuzo stade ikura ingofero video mu gitaramo kwizihiza umunsi mpuzamahanga w ururimi kavukire umuzungu yavuze ikinyarwanda benshi baratangara ariko ahanini batangazwa n inyota afite kumeny"],
]

DESCRIPTION = (
    "Paste a Kinyarwanda news article (title and text). Two models predict one of 14 topics: "
    "politics, sport, economy, health, entertainment, history, technology, tourism, culture, "
    "fashion, religion, environment, education, relationship.\n\n"
    "Model 1 is a baseline (TF-IDF character n-grams with a linear SVM). Model 2 is AfroXLMR fine tuned "
    "on the KINNEWS dataset (Niyongabo et al., 2020). On my test set the baseline scored slightly "
    "higher overall, so showing both lets you see where they agree and where they differ. "
    "Small topics such as history and environment are the least reliable. "
    "The examples are the beginnings of articles from the KINNEWS test set."
)

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=8, label="Kinyarwanda news text"),
    outputs=[
        gr.Label(num_top_classes=5, label="Fine tuned AfroXLMR"),
        gr.Textbox(label="Baseline (TF-IDF and linear SVM)"),
        gr.Textbox(label="Comparison"),
    ],
    title="Kinyarwanda news topic classifier",
    description=DESCRIPTION,
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
