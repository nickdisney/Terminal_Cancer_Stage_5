import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# ai_models/natural_language_understanding/nlu_model.py

from utils.tokenizer import get_tokenizer

class NLUModel:
    def __init__(self, model_name):
        self.tokenizer = get_tokenizer(model_name)
        # ... (existing initialization code)

    def preprocess_text(self, text):
        # Tokenize and preprocess the input text using the tokenizer
        tokens = self.tokenizer.tokenize(text)
        # Perform additional preprocessing if needed
        return tokens


class NLUModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.intent_classifier = None

    def train_intent_classifier(self, labeled_data):
        X = [data["text"] for data in labeled_data]
        y = [data["intent"] for data in labeled_data]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("svm", SVC(kernel="linear"))
        ])

        pipeline.fit(X_train, y_train)
        self.intent_classifier = pipeline

        y_pred = self.intent_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Intent Classification Accuracy: {accuracy}")

    def predict_intent(self, text):
        if self.intent_classifier is None:
            raise ValueError("Intent classifier not trained. Call train_intent_classifier() first.")

        intent = self.intent_classifier.predict([text])[0]
        return intent

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        return entities