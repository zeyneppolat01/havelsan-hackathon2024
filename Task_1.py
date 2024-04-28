import os
import pandas as pd
import re
import nltk
from nltk import word_tokenize, SnowballStemmer
from snowballstemmer import TurkishStemmer
import zeyrek
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score


nltk.download('punkt')


def load_data(directory):
    texts, labels = [], []
    categories = os.listdir(directory)
    for category in categories:
        category_dir = os.path.join(directory, category)
        files = os.listdir(category_dir)
        for file in files:
            file_path = os.path.join(category_dir, file)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(category)
    return pd.DataFrame({'metin': texts, 'kategori': labels})


def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r'[^a-zğüşöçıİĞÜŞÖÇ\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return stopwords


def preprocess_data(data, stopwords_path):
    turkish_stopwords = load_stop_words(stopwords_path)
    stemmer = TurkishStemmer()
    data['metin'] = data['metin'].apply(clean_text)
    data['metin'] = data['metin'].apply(
        lambda x: ' '.join([stemmer.stemWord(token) for token in word_tokenize(x) if token not in turkish_stopwords]))
    return data



directory = 'X:\\sınıflandırma_train_data\\news'
stopwords_path = 'X:\\turkce-stop-words.txt'
training_data = preprocess_data(load_data(directory), stopwords_path)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_data['metin'])
y = training_data['kategori']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)


scaler_ros = StandardScaler(with_mean=False)  # TF-IDF için with_mean=False kullanılır
X_train_scaled_ros = scaler_ros.fit_transform(X_train_ros)
X_test_scaled_ros = scaler_ros.transform(X_test)


rf_model_ros_sample = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_ros_sample.fit(X_train_scaled_ros, y_train_ros)

print("\n---------- Training Complete ------------\n")

# Load and preprocess real test data
real_test_data_path = 'X:\\siniflandirma.csv'
real_test_data = pd.read_csv(real_test_data_path, delimiter='|', names=['metin', 'kategori'], header=None, skiprows=1)
real_test_data = preprocess_data(real_test_data, stopwords_path)

# Vectorize real test data using the same vectorizer
real_test_vectorized = vectorizer.transform(real_test_data['metin'])

# Apply the same scaling as was applied to the training data
real_test_scaled = scaler_ros.transform(real_test_vectorized)

# Predict categories for real test data
real_test_predictions = rf_model_ros_sample.predict(real_test_scaled)

# Optionally, save or print the predictions
real_test_data['predicted_kategori'] = real_test_predictions
print(real_test_data[['metin', 'predicted_kategori']])

# Save the results to a new CSV if needed
real_test_data.to_csv('X:\\siniflandirma_results.csv', index=False)
