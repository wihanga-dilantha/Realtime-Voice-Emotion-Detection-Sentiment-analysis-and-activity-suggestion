import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.header("Sentiment Analysis", divider=True)

activity = {
    'positive': 'Group activities, Outdoor activities, Social events',
    'negative': 'Relaxation and mindfulness, One on one activities, Comfort Activities',
    'neutral': 'Educational activities, Health and fitness activities, Creative expression'
}

def sentiment_analysis(result_container, result_container1, activity_container, chart_container,activity_title_container,chart_title_container,chart_devider_container,activity_divider_container):
    # Load pre-trained model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Function to analyze sentiment of a text
    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    # Read transcriptions from file and perform sentiment analysis
    positive_scores = []
    negative_scores = []
    
    with st.spinner('Analyzing...'):
        with open("transcriptions.txt", "r", encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:
                    probs = analyze_sentiment(line)
                    positive_prob = probs.squeeze()[1].item()  # Probability of positive class (index 1)
                    negative_prob = probs.squeeze()[0].item()  # Probability of negative class (index 0)
                    
                    # Store the probabilities
                    positive_scores.append(positive_prob)
                    negative_scores.append(negative_prob)

        # Calculate percentages
        total_count = len(positive_scores)
        positive_count = sum(1 for score in positive_scores if score > 0.5)
        negative_count = sum(1 for score in negative_scores if score > 0.5)

        if total_count > 0:
            positive_percentage = (positive_count / total_count) * 100
            negative_percentage = (negative_count / total_count) * 100
        else:
            positive_percentage = 0
            negative_percentage = 0

        result_container.markdown(f"Positive Sentiment Percentage: {positive_percentage:.2f}%")
        result_container1.markdown(f"Negative Sentiment Percentage: {negative_percentage:.2f}%")

        if positive_percentage > negative_percentage:
            suggest = activity['positive']
        elif positive_percentage < negative_percentage:
            suggest = activity['negative']
        else:
            suggest = activity['neutral']

        activity_title_container.markdown("## Activity Suggestions: ")
        activity_divider_container.markdown("---")
        activity_container.markdown(suggest)

    # Plot the sentiment scores
    chart_title_container.markdown("## Sentiment Chart")
    chart_devider_container.markdown("---")
    fig, ax = plt.subplots()
    ax.plot(positive_scores, label='Positive')
    ax.plot(negative_scores, label='Negative')
    ax.set_xlabel('Line Number')
    ax.set_ylabel('Sentiment Score')
    ax.legend()
    chart_container.pyplot(fig)

    st.success("Done")

result_container = st.empty()
result_container1 = st.empty()

activity_title_container = st.empty()
activity_divider_container = st.empty()
activity_container = st.empty()

chart_title_container = st.empty()
chart_devider_container = st.empty()
chart_container = st.empty()

left, middle, right = st.columns([10, 1, 1], vertical_alignment="bottom")

start_button = middle.button("Start")

if start_button:
    sentiment_analysis(result_container, result_container1, activity_container, chart_container,activity_title_container,chart_title_container,chart_devider_container,activity_divider_container)
