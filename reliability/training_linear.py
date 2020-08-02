import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")
df.set_index("Unnamed: 0")

y = df.label
df.drop("label", axis=1)


# @todo add analysis for test/train split
X_train, X_test, y_train, y_test = train_test_split (
    df['text'], y, test_size=0.33, random_state=53
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
vectorizer_filename = 'tfidf_vectorizer.pk'
pickle.dump(tfidf_vectorizer, open(vectorizer_filename,'wb'))

# @todo: parameter tuning
linear_clf = PassiveAggressiveClassifier(
    n_iter_no_change=50)

linear_clf.fit(tfidf_train, y_train)

filename = 'linear_clf_model.pk'
pickle.dump(linear_clf, open(filename, 'wb'))
