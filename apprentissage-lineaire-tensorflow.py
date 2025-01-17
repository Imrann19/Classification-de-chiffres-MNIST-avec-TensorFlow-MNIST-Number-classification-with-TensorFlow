import tensorflow as tf
import numpy as np
import random

#Chargement des donnés MNIST / Loading MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0 #Normalisation des donnés / Normalizing the data 

#Construction du modèle / Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(10, activation='softmax')
])

#Compilation du modèle / Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entraînement du modèle...")
model.fit(X_train, y_train, epochs=5)#Entrainement du modèle / Training of the model
print("Entraînement terminé.")

#Evaluation du modèle / Evaluating the model
test_FP, test_P = model.evaluate(X_test, y_test)
print(f"\nPrécision sur les données de test : {test_P * 100:.2f}%")

sample_index = random.randint(0, 9)
sample = X_test[sample_index].reshape(1, 28, 28)  

predictions = model.predict(sample)

predicted_class = np.argmax(predictions)  

#Affichage du résultat / Displaying results
print(f"\nLe modèle prédit que ce chiffre est : {predicted_class}")
print(f"Le chiffre réel est : {y_test[sample_index]}")
