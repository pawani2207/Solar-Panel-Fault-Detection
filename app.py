
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Solar Panel Fault Detection",
    page_icon="☀️",
    layout="centered"
)

# App title
st.title("☀️ Solar Panel Fault Detection")
st.write(
    "Upload a solar panel image to identify its condition "
    "and get a maintenance recommendation."
)

# Load trained model and class names
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(
        "solar_panel_6class_model.keras"
    )

    with open("class_names.json", "r") as file:
        class_names = json.load(file)

    return model, class_names


try:
    model, class_names = load_model_and_classes()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# Maintenance recommendations
maintenance_advice = {
    "Clean": (
        "Routine inspection as per the site's maintenance schedule."
    ),
    "Dusty": (
        "Consider a cleaning inspection within 7 days."
    ),
    "Bird-drop": (
        "Consider a cleaning inspection within 1–3 days."
    ),
    "Snow-Covered": (
        "Check promptly and follow site safety procedures."
    ),
    "Physical-Damage": (
        "Arrange a prompt inspection by a qualified technician."
    ),
    "Electrical-damage": (
        "Arrange prompt professional inspection. "
        "Do not handle potentially damaged electrical equipment."
    )
}


# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload a solar panel image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
    image,
    caption="Uploaded Solar Panel",
    width=300
)
    

    if st.button("🔍 Detect Fault", use_container_width=True):

        with st.spinner("Analyzing solar panel image..."):
            resized_image = image.resize((224, 224))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        # Display prediction
        st.subheader("📊 Detection Result")

        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Display maintenance recommendation
        st.subheader("🛠️ Suggested Maintenance")

        advice = maintenance_advice.get(
            predicted_class,
            "Manual inspection is recommended."
        )

        st.info(advice)

        # Display all class scores
        with st.expander("View scores for all classes"):
            for i, class_name in enumerate(class_names):
                st.write(
                    f"{class_name}: "
                    f"{predictions[i] * 100:.2f}%"
                )

        st.caption(
            "Note: Maintenance suggestions are rule-based examples, "
            "not validated predictions of the exact maintenance date. "
            "Actual timing depends on site conditions and professional "
            "inspection. Confidence is not guaranteed accuracy."
        )

st.divider()
st.caption("Solar Panel Fault Detection | CNN-Based Prototype")
