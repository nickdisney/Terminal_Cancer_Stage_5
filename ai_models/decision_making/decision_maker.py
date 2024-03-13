from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionMaker:
    def __init__(self):
        self.decision_tree = None

    def train_decision_tree(self, labeled_data):
        X = [data["state"] for data in labeled_data]
        y = [data["action"] for data in labeled_data]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.decision_tree = DecisionTreeClassifier()
        self.decision_tree.fit(X_train, y_train)

        y_pred = self.decision_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Decision Tree Accuracy: {accuracy}")

    def make_decision(self, state):
        if self.decision_tree is None:
            raise ValueError("Decision tree not trained. Call train_decision_tree() first.")

        action = self.decision_tree.predict([state])[0]
        return action