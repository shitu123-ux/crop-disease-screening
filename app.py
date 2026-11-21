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
    "Rust": {
        "en": "Rust",
        "ha": "Tsatsa"
    },
    "Blight": {
        "en": "Blight",
        "ha": "Bushewar Ganye"
    }
}

# Risk levels
RISK_LEVELS = {
    "Healthy": {
        "en": "🟢 Low Risk",
        "ha": "🟢 Hadari Kadan"
    },
    "Spot": {
        "en": "🟡 Medium Risk",
        "ha": "🟡 Matsakaicin Hadari"
    },
    "Rust": {
        "en": "🟡 Medium Risk",
        "ha": "🟡 Matsakaicin Hadari"
    },
    "Blight": {
        "en": "🔴 High Risk",
        "ha": "🔴 Hadari Mai Tsanani"
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

    "Rust": {
        "en": "Rust disease detected. Remove infected leaves and apply a recommended fungicide. Avoid overhead irrigation where possible.",
        "ha": "An gano cutar tsatsa. A cire ganyen da suka kamu da cuta sannan a yi amfani da maganin fungus da ya dace. A guji yayyafa ruwa kai tsaye a kan ganye idan zai yiwu."
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

    # Calibration
    t = CAL.get("temperature", 1.0)

    e = np.exp(logits / t)
    probs = e / np.sum(e)

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
         "ha": "Babu shawara a yanzu."}
    )

    risk = RISK_LEVELS.get(
        best_label,
        {"en": "Unknown", "ha": "Ba a sani ba"}
    )

    if language == "Hausa":

        diagnosis = f"""
# 🌿 Sakamako

## Cuta
**{disease_name['ha']}**

## Kwarin Gwiwa
**{confidence:.1f}%**

## Matsayin Hadari
**{risk['ha']}**

## Shawara
{treatment['ha']}

---

### Ma'anar Matsayin Hadari

🟢 Healthy (Low Risk)

🟡 Spot (Medium Risk)

🟡 Rust (Medium Risk)

🔴 Blight (High Risk)
"""

    else:

        diagnosis = f"""
# 🌿 Diagnosis

## Disease
**{disease_name['en']}**

## Confidence
**{confidence:.1f}%**

## Risk Level
**{risk['en']}**

## Recommendation
{treatment['en']}

---

### Risk Legend

🟢 Healthy (Low Risk)

🟡 Spot (Medium Risk)

🟡 Rust (Medium Risk)

🔴 Blight (High Risk)
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

### Risk Legend

🟢 Healthy (Low Risk)

🟡 Spot (Medium Risk)

🟡 Rust (Medium Risk)

🔴 Blight (High Risk)
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
        num_top_classes=4,
        label="Top Predictions"
    )

    # Automatic prediction
    img.change(
        predict,
        inputs=[img, language],
        outputs=[result, probabilities]
    )

    analyse_btn = gr.Button("Analyse Leaf")

    analyse_btn.click(
        predict,
        inputs=[img, language],
        outputs=[result, probabilities]
    )

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
