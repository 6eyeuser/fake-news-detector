import streamlit as st
from transformers import pipeline

classifier = pipeline("text-classification", model="model")

st.title("Fake News Detector")

text = st.text_area("Enter news text")

if st.button("Check"):
    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    if label == "LABEL_1":
        st.success(f"Real News ✅ (Confidence: {score:.2f})")
    else:
        st.error(f"Fake News ❌ (Confidence: {score:.2f})")