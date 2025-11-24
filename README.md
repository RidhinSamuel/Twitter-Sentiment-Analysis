## Dataset
---

## 🔧 Step-by-Step Sentiment Analysis Project (on 1.6M Tweets)

---

### ✅ Step 1: **Import libraries**

```python
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```

---

### ✅ Step 2: **Load the dataset**

```python
df = pd.read_csv('your_dataset.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# We'll use only 'target' and 'text'
df = df[['target', 'text']]
```

---

### ✅ Step 3: **Convert target values**

```python
df['target'] = df['target'].replace({0: 'negative', 2: 'neutral', 4: 'positive'})
```

---

### ✅ Step 4: **Clean the tweets**

```python
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # remove links
    text = re.sub(r"@\w+", "", text)             # remove mentions
    text = re.sub(r"#", "", text)                # remove hashtags symbol
    text = re.sub(r"[^\w\s]", "", text)          # remove punctuation
    text = " ".join(word for word in text.split() if word not in stopwords.words('english'))
    return text

df['clean_text'] = df['text'].apply(clean_text)
```

---

### ✅ Step 5: **Split into train/test**

```python
X = df['clean_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### ✅ Step 6: **Vectorize the text**

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

---

### ✅ Step 7: **Train a model**

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
```

---

### ✅ Step 8: **Evaluate the model**

```python
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
```

---

### ✅ Step 9: **Make predictions**

```python
def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# Example
print(predict_sentiment("I love this product"))
```

---

