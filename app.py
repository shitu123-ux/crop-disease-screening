import gradio as gr
import numpy as np
import onnxruntime as ort
import json
import os
from PIL import Image

# Load labels
LABELS = [l.strip() for l in open("labels.txt") if l.strip()]

# Load calibration settings
CAL = json.load(open("calibration.json"))

# Load model
SESSION = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

# Disease names
CLASS_NAMES = {
    "Healthy": {
        "en": "Healthy",
        "ha": "Lafiyayyen Ganye"
    },
    "Spot": {
        "en": "Leaf Spot",
        "ha": "Tabon Ganye"
    },
    "Blight": {
        "en": "Blight",
        "ha": "Bushewar Ganye"
    }
}

# Treatment recommendations
TREATMENTS = {
    "Healthy": {
        "en": "The plant appears healthy. Continue proper watering, fertilization and regular monitoring.",
        "ha": "Shukar tana da lafiya. A ci gaba da ba ta ruwa, taki da kuma kula da ita akai-akai."
    },

    "Spot": {
        "en": "Remove infected leaves. Improve air circulation around plants and apply a suitable fungicide if necessary.",
        "ha": "A cire ganyen da suka kamu da cuta. A tabbatar iska na shiga tsakanin tsire-tsire sannan a yi amfani da maganin fungus idan ya zama dole."
    },

    "Blight": {
        "en": "Remove heavily infected leaves immediately. Apply a recommended fungicide and avoid overwatering.",
        "ha": "A cire ganyen da suka kamu sosai da cuta. A yi amfani da maganin fungus da ake ba da shawara sannan a guji yawan ruwa."
    }
}


def predict(img, language):

    if img is None:
        if language == "Hausa":
            return "📷 Dauki hoton ganye ko ka loda shi.", {}
        else:
            return "📷 Upload or capture a leaf image.", {}

    # Preprocess image
    img = img.convert("RGB").resize((224, 224))

    x = np.asarray(img).astype("float32") / 255.0

    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    x = x.transpose(2, 0, 1)[None].astype("float32")

    # Run inference
    logits = SESSION.run(None, {"image": x})[0]

    # Temperature scaling
    t = CAL.get("temperature", 1.0)

    e = np.exp(logits / t)
    probs = e / e.sum()

    # Top 3 predictions
    top = np.argsort(probs[0])[::-1][:3]

    best_label = LABELS[top[0]]
    confidence = float(probs[0, top[0]]) * 100

    disease_name = CLASS_NAMES.get(
        best_label,
        {"en": best_label, "ha": best_label}
    )

    treatment = TREATMENTS.get(
        best_label,
        {"en": "No recommendation available.",
         "ha": "Babu shawara a halin yanzu."}
    )

    # Language-specific output
    if language == "Hausa":

        diagnosis = f"""
# 🌿 Sakamako

## Cuta
**{disease_name['ha']}**

## Kwarin Gwiwa
**{confidence:.1f}%**

## Shawara
{treatment['ha']}
"""

    else:

        diagnosis = f"""
# 🌿 Diagnosis

## Disease
**{disease_name['en']}**

## Confidence
**{confidence:.1f}%**

## Recommendation
{treatment['en']}
"""

    top3 = {}

    for i in top:

        label = LABELS[i]

        if language == "Hausa":
            display_label = CLASS_NAMES.get(
                label,
                {"ha": label}
            )["ha"]
        else:
            display_label = CLASS_NAMES.get(
                label,
                {"en": label}
            )["en"]

        top3[display_label] = float(probs[0, i])

    return diagnosis, top3


with gr.Blocks(title="Crop Disease Screening System") as demo:

    gr.Markdown("""
# 🌿 Crop Disease Screening System

Detect crop diseases using uploaded images or your phone camera.

Gano cututtukan amfanin gona ta amfani da hoton ganye ko kyamarar waya.
""")

    language = gr.Dropdown(
        choices=["English", "Hausa"],
        value="English",
        label="Language / Harshe"
    )

    img = gr.Image(
        sources=["upload", "webcam"],
        type="pil",
        label="Upload or Scan Leaf"
    )

    result = gr.Markdown()

    probabilities = gr.Label(
        num_top_classes=3,
        label="Top Predictions"
    )

    # Auto prediction when image changes
    img.change(
        predict,
        inputs=[img, language],
        outputs=[result, probabilities]
    )

    # Manual button
    analyse_btn = gr.Button("Analyse Leaf")

    analyse_btn.click(
        predict,
        inputs=[img, language],
        outputs=[result, probabilities]
    )

# Render deployment
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
