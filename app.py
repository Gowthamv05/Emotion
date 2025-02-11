import warnings
from transformers import pipeline
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_models():
    print("Loading the models...")
    sentiment_model = pipeline("sentiment-analysis")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    print("Models loaded successfully!")
    return sentiment_model, emotion_model

def transform_to_assertive(emotion, sentence):
    """
    Provide a professional/assertive version based on detected emotion and sentence.
    """
    transformations = {
        "anger": "I understand this happened and will address it constructively.",
        "sadness": "I acknowledge the results and will take steps to improve.",
        "joy": "This is a great moment, and I appreciate it.",
        "neutral": "I acknowledge this event and will proceed accordingly."
    }
    
    # Use predefined transformations based on detected emotion
    assertive_sentence = transformations.get(emotion.lower(), sentence)
    return assertive_sentence

def analyze_and_transform_sentence(sentence, sentiment_model, emotion_model):
    """
    Analyze sentiment and transform the sentence into an assertive tone.
    """
    sentiment_results = sentiment_model(sentence)
    emotion_results = emotion_model(sentence)
    detected_emotion = emotion_results[0]['label'].lower()
    
    # Use custom rules for assertive transformation based on emotion
    transformed_sentence = transform_to_assertive(detected_emotion, sentence)
    
    return detected_emotion, transformed_sentence

# Streamlit app
def main():
    st.title("Sentiment Analysis and Assertive Transformation")
    st.write("Enter a sentence to analyze its emotion and transform it into an assertive tone.")

    # Input box for the sentence
    input_sentence = st.text_input("Enter a sentence:", "")
    
    if st.button("Run Analysis"):
        if input_sentence:
            sentiment_model, emotion_model = load_models()
            detected_emotion, transformed_result = analyze_and_transform_sentence(input_sentence, sentiment_model, emotion_model)
            
            st.write("\n--- Results ---")
            st.write(f"**Input Sentence:** \"{input_sentence}\"")
            st.write(f"**Detected Emotion:** {detected_emotion.capitalize()}")
            st.write(f"**Professional/Assertive Sentence:** \"{transformed_result}\"")
        else:
            st.warning("Please enter a sentence to analyze.")

if __name__ == '__main__':
    main()
