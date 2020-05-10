from catboost import CatBoostRegressor
from flask import Flask , jsonify , request
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from flask_cors import CORS
from googletrans import Translator
app=Flask(__name__)
CORS(app)
cors=CORS(app,resources={r"/*":{"origins":"*"}})
@app.route('/<string:ch>')
def index(ch):
  original=ch
  ch,lang=translation(ch)
  translated=ch
  model=CatBoostRegressor()
  model.load_model('m2.cbm',format='cbm')
  ch=lem(points(remove(ch)))
  with open('vectorizer.pk', 'rb') as handle:
    tfidf = pickle.load(handle)
  ch=' '.join(ch)
  x= tfidf.transform([ch]).toarray()
  r=model.predict(x)[0]
  #r=TextBlob(translated).sentiment.polarity
  return(jsonify({'sentement':r,'language':lang,'original':original,'translated':translated}))
def remove(ch):
  nltk.download("stopwords")
  stop=nltk.corpus.stopwords.words('english')
  ch=str(ch)
  l=ch.split()
  l2=[]
  for i in l:
    if (i.lower() not in stop) and  ("@" not in i.lower()) and  ("http" not in i.lower()):
      l2.append(i.lower())
  return(l2)
def points(l):
  l2=[]
  for i in l:
    table = str.maketrans('', '', string.punctuation)
    l2.append(i.translate(table))
  return(l2)
def lem(l):
  nltk.download('wordnet')
  Lemmatizer = WordNetLemmatizer()
  l=[Lemmatizer.lemmatize(word) for word in l]
  return(l)
def translation(c,target="en-US"):
  translator = Translator()
  ch=translator.translate(c).text
  lang=translator.translate(c).src
  return(ch,lang)
if __name__=='__main__':
	app.run(debug=True)