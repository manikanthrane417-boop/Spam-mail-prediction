import streamlit as st
import pickle

def load_model():
    with open ("model.pkl","rb") as f:
        model=pickle.load(f)

    with open ("vectorer.pkl","rb") as f:
        vectorizer=pickle.load(f)
    return model,vectorizer

def predict(text):
    model,vectorizer=load_model()
    X=vectorizer.transform([text])
    result=model.predict(X)
    if result[0]==1:
        return "Spam"
    else:
        return  "Not spam"

def main():
    st.title("Spam Mail Prediction")
    st.write("Enter the messsage to predict whether it is spam or not")
    msg=st.text_area("Message")
    result=""
    if st.button("Predict"):
        result=predict(msg)
    st.success(f"The message is: {result}")

if __name__=="__main__":
    main()