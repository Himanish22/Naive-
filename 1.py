import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('iris.csv')
print(data.columns)

# Separate features and target
X = data.drop('Species', axis=1)
y = data['Species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
err_rate = 1 - acc
report = classification_report(y_test, y_pred)

# Display metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Error Rate: {err_rate * 100:.2f}%")

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens",
            xticklabels=data['Species'].unique(), yticklabels=data['Species'].unique())
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
