import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('EnjoySport.csv')

# Encode categorical features
encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

# Separate features and target
X = data.drop('PlayTennis', axis=1)
y = data['PlayTennis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Metrics
conf_matrix = confusion_matrix(y_test, predictions)
acc = accuracy_score(y_test, predictions)
err_rate = 1 - acc
report = classification_report(y_test, predictions)

# Display metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Error Rate: {err_rate * 100:.2f}%")

# Plot confusion matrix
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples",
            xticklabels=data['PlayTennis'].unique(), yticklabels=data['PlayTennis'].unique())
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
