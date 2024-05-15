import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
nltk.download('stopwords')

Loading and Reading the Dataset

df=pd.read_csv("20191226-reviews.csv")
df.head()

df.drop('asin',axis=1,inplace=True)
df.drop('name',axis=1,inplace=True)
df.head()

df.dropna(inplace=True)

Generating Longest Review by Length

df['length'] = df['body'].apply(len)

Let's extract the largest review...

df[df['verified']=='True'][['body','length']].sort_values(by='length',ascending=False).head().body

Text Pre-Processing and Generating Bag-of-Words

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer

bow_transformer.fit(df['body'])
print("Total Vocabulary:",len(bow_transformer.vocabulary_))

review4 = df['body'][3]
review4

bow_msg4 = bow_transformer.transform([review4])
print(bow_msg4)
print(bow_msg4.shape)

bow_reviews = bow_transformer.transform(df['body'])

print("Shape of Bag of Words Transformer for the entire reviews corpus:",bow_reviews.shape)
print("Amount of non zero values in the bag of words model:",bow_reviews.nnz)

print("Sparsity:",np.round((bow_reviews.nnz/(bow_reviews.shape[0]*bow_reviews.shape[1]))*100,2))

tfidf_transformer = TfidfTransformer().fit(bow_reviews)
tfidf_rev4 = tfidf_transformer.transform(bow_msg4)
print(bow_msg4)

tfidf_reviews = tfidf_transformer.transform(bow_reviews)
print("Shape:",tfidf_reviews.shape)
print("No. of Dimensions:",tfidf_reviews.ndim)
