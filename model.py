from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pickle


digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='micro')}")

with open("classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
