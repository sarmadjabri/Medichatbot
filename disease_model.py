import pandas as pd
from sklearn.preprocessing import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("dataset.csv")

# Preprocess data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["symptoms"])

X = tokenizer.texts_to_matrix(df["symptoms"], mode="binary")
y = pd.get_dummies(df["disease"]).values

# Train
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X.shape[1],)))
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save
model.save("disease_model.h5")

# Load
loaded_model = load_model("disease_model.h5")

# Use loaded model for prediction
new_symptoms = ["fever", "headache", "cough"]
new_symptoms = tokenizer.texts_to_matrix([new_symptoms], mode="binary")
prediction = loaded_model.predict(new_symptoms)

print("Prediction:", prediction)
