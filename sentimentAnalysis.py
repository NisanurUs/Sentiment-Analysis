# The necessary libraries are being imported. pandas is used for data manipulation,
# nltk for text preprocessing, and sklearn for model building and evaluation.
# To address SMOTE data imbalance, RandomForestClassifier and 
# GradientBoostingClassifier are used for alternative models.
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# upload the dataset where located "tweets.csv" for education of algorithm 
df = pd.read_csv('tweets.csv')       

# Downloaded The required nltk datasets. Stopwords is used for English stop 
# words (e.g. "the", "is") and punt is used for word tokenization.
nltk.download('stopwords')
nltk.download('punkt')

# This function converts text to lowercase, removes punctuation, 
# separates words into tokens, and removes stop words.
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# The preprocess_text function is applied to each text in the data set and 
# the results are saved in the cleaned_text column.
df['cleaned_text'] = df['text'].apply(preprocess_text)

# maximum of 5000 features will be extracted, and ngram_range=(1, 2) indicates
# that n-grams of 1-2 words will be used.
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_text'])

# The emotional states of the texts (positive, negative, neutral) are tagged and
# these tags are converted into numerical values.
# These labels are assigned to the y variable.
df['label'] = df['airline_sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})
y = df['label']

# Data imbalance is eliminated using SMOTE (Synthetic Minority Over-sampling Technique).
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# The data is divided into training and test sets. test_size=0.2 indicates that 
# 20% of the data will be used as the test set.
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# LogisticRegression model is trained with balanced class weights. max_iter=200 specifies 
# the maximum number of iterations.
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train, y_train)

# The model makes predictions on the test set and the results are evaluated. 
# With accuracy_score and classification_report, the accuracy and other metrics 
# (precision, recall, f1-score) of the model are calculated.
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Two alternative models, RandomForestClassifier and GradientBoostingClassifier,
# are trained and evaluated on the test set, and the results are printed.
# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_y_pred))
print(classification_report(y_test, gb_y_pred))

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}
# The best hyperparameters for RandomForestClassifier are found using GridSearchCV.
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
# The best model is saved as best_model.
print("Best parameters:", grid.best_params_)
best_model = grid.best_estimator_

# A function is being created that predicts the mood of new dataset. 
# This function cleans the text, vectorizes it, and makes predictions with the trained model.
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = best_model.predict(text_vector)
    sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    return sentiment_map[prediction[0]]

# Reading comments.csv 
comments_df = pd.read_csv('comments.csv')

# apply the sentiment analzy
comments_df['sentiment'] = comments_df['text'].apply(predict_sentiment)

# write results to sentiment_results.csv
comments_df.to_csv('sentiment_results.csv', index=False)

