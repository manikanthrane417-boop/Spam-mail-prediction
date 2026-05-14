import pickle

def load_model():
    with open ("spam_mail_prediction/model.pkl","rb") as f:
        model=pickle.load(f)

    with open ("spam_mail_prediction/vectorer.pkl","rb") as f:
        vectorizer=pickle.load(f)
    return model,vectorizer

def predict(text):
    model,vectorizer=load_model()
    X=vectorizer.transform([text])
    result=model.predict(X)
    return "Spam" if result[0]==1 else "Not spam"

