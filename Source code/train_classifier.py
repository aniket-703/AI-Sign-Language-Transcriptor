import os
import pickle 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')
data_path = os.path.join(MODEL_DIR, 'data_final.pickle')

if not os.path.exists(data_path):
    print(f"Dataset file '{data_path}' not found.")
    exit(1)

with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

# Process features to include squared terms (84 features)
processed_data = []
for seq in data_dict['data']:
    item = list(seq)
    if len(item) == 42:
        item.extend([x**2 for x in item])
    elif len(item) < 84:
        item = item + [0] * (84 - len(item))
    elif len(item) > 84:
        item = item[:84]
    processed_data.append(item)

data = np.asarray(processed_data)
labels = np.asarray(data_dict['labels'])

print(f"Training dataset shape: {data.shape}, Total samples: {len(labels)}")

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"Accuracy: {score * 100:.2f}% of test samples classified correctly!")

model_path = os.path.join(MODEL_DIR, 'model_final.p')
with open(model_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Model saved successfully to: {model_path}")
