# model_training.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample data
data = {
    'experience': [1, 2, 3, 4, 5],
    'education_level': [1, 2, 2, 3, 3],  # 1: High School, 2: Bachelor, 3: Master
    'salary': [30000, 35000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['experience', 'education_level']]
y = df['salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
