import streamlit as st
import requests as rq
import pickle 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
def transform_text(text: str):
    text = text.lower()
    text = word_tokenize(text)
    text_without_punctuation = []
    for word in text:
        if word.isalnum():
            text_without_punctuation.append(word)
    text = text_without_punctuation[:]
    text_without_punctuation.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            text_without_punctuation.append(i)
    ps = PorterStemmer()
    text = text_without_punctuation[:]
    text_without_punctuation.clear()
    for i in text:
        text_without_punctuation.append(ps.stem(i))
    return " ".join(text_without_punctuation)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('mnb.pkl','rb'))
st.title("Email Spam Classifier")
txt = st.text_input("Enter the message")
if st.button("Predict"):
    # 1. preprocess
    transformed_txt = transform_text(txt)
    # 2. vectorize
    vectorized_txt = tfidf.transform([transformed_txt])
    # 3. predict
    prediction = model.predict(vectorized_txt)[0]
    # 4. display
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

