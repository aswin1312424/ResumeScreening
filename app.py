from flask import Flask,render_template,request
from PyPDF2 import PdfReader
import re
import pickle
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()


app=Flask(__name__)


def corpus_cleaning(x):
    text=x.lower()
    text=re.sub("(www|http:|https:)+[^\s]+[\w]"," ",text)
    text=re.sub("[^a-z0-9]"," ",text)
    text=[lem.lemmatize(word,pos="v") for word in text.split() if word not in stopwords.words("english")]
    return " ".join(text)

def words_to_vectors(x):
    with open("Notebooks/tfidf.pkl","rb") as f:
        vectorizer=pickle.load(f)
    vectors=vectorizer.transform([x]).toarray()
    return vectors

def prediction(x):
    with open("Notebooks/model.pkl","rb") as f:
        model=pickle.load(f)
    with open("Notebooks/label_encoder.pkl","rb") as f:
        label_encoder=pickle.load(f)
    return label_encoder.classes_[model.predict(x)[0]]

@app.route("/",methods=["GET","POST"])
def file_upload():
    if request.method=="GET":
        return render_template("homepage.html")
    resume=request.files["upload_file"]
    if resume:
        reader=PdfReader(resume)
        text=""
        for page in reader.pages:
            text+=page.extract_text()

        cleaned_text=corpus_cleaning(text)
        vectors=words_to_vectors(cleaned_text)
        predicted=prediction(vectors)
        return render_template("homepage.html",role=predicted)
    else:
        return "No File Uploaded"


if __name__=="__main__":
    app.run()
