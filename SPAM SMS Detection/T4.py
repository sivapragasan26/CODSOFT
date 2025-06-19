import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Dataset (and fix column mismatch)
df = pd.read_csv("T4.csv", encoding='ISO-8859-1')

# Use only the first 2 relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 2: Encode Labels: ham → 0, spam → 1
df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

# Step 6: Train and evaluate
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
