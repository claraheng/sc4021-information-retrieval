import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Step 1: Synthetic EV Dataset (expand this in production)
data = {
    'text': [
        # Neutral (facts)
        "The Tesla Model 3 has a range of 300 miles per charge.",
        "EVs use lithium-ion batteries for energy storage.",
        "Charging an EV at home takes 8-10 hours on a Level 2 charger.",
        "The Nissan Leaf weighs 3,500 pounds.",
        "EVs produce zero tailpipe emissions.",
        "Regenerative braking recaptures energy in EVs.",
        "The average EV battery capacity is 60-100 kWh.",
        "Tesla Superchargers deliver up to 250 kW.",
        "EVs have fewer moving parts than gas vehicles.",
        "The Chevy Bolt has a top speed of 93 mph.",
        "EV sales reached 10% of total vehicles in 2023.",
        "Battery degradation is typically 2% per year.",
        "EVs qualify for federal tax credits up to $7,500.",
        "Wireless charging is emerging for EVs.",
        "The Ford Mustang Mach-E has AWD options.",
        "EVs use DC fast charging for quick stops.",
        "Porsche Taycan accelerates 0-60 in 2.6 seconds.",
        "EV infrastructure includes over 100,000 public chargers in the US.",
        "Batteries in EVs last 8-10 years.",
        "Hyundai Ioniq 5 supports 350 kW charging.",
        
        # Opinionated (thoughts)
        "EVs are way better than gas guzzlers and save the planet!",
        "I hate how long it takes to charge my EV; gas is faster.",
        "Tesla is overrated; their build quality sucks.",
        "Switching to EVs was the best decision ever.",
        "EVs are a scam pushed by governments.",
        "Love my Rivian; it's the coolest truck out there.",
        "Charging stations are everywhere now—EVs rock!",
        "EVs feel sluggish compared to sports cars.",
        "I'm obsessed with my Polestar; silent and smooth.",
        "Gas cars will never die; EVs are just a fad.",
        "EVs make driving fun with instant torque.",
        "Too expensive upfront, but EVs pay off long-term.",
        "My EV changed my life; no more oil changes!",
        "Range anxiety ruins EVs for road trips.",
        "Lucid Air is the luxury EV king.",
        "EVs are boring to drive without engine noise.",
        "Fully committed to EVs; fossil fuels are done.",
        "Hate the tiny trunks in most EVs.",
        "EVs are the future; get on board now!",
        "Still prefer the roar of a V8 over EV silence."
    ],
    'label': ['Neutral']*20 + ['Opinionated']*20
}

df = pd.DataFrame(data)
print("Dataset preview:")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")

# Step 2: Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# Step 3: Train-test split
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 4: Pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1,2))),
    ('clf', LogisticRegression(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Predict new texts
new_texts = [
    "The Rivian R1T tows 11,000 pounds.",  # Neutral
    "EVs are garbage and unreliable."       # Opinionated
]
new_clean = [preprocess_text(t) for t in new_texts]
predictions = pipeline.predict(new_clean)
probabilities = pipeline.predict_proba(new_clean)

for text, pred, prob in zip(new_texts, predictions, probabilities):
    print(f"\nText: '{text}'")
    print(f"Prediction: {pred} (Neutral: {prob[0]:.2f}, Opinionated: {prob[1]:.2f})")
