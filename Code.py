import tensorflow as tf
import numpy as np
import random

#Chargement des donnes MNIST / Loading MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0 #Normalisation des donnes / Normalizing the data 

#Construction du modele / Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(10, activation='softmax')
])

#Compilation du modele / Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entraînement du modèle...")
model.fit(X_train, y_train, epochs=5)#Entrainement du modele / Training of the model
print("Entrainement termine.")

#Evaluation du modele / Evaluating the model
test_FP, test_P = model.evaluate(X_test, y_test)
print(f"\nPrecision sur les donnees de test : {test_P * 100:.2f}%")

sample_index = random.randint(0, 9)
sample = X_test[sample_index].reshape(1, 28, 28)  

predictions = model.predict(sample)

predicted_class = np.argmax(predictions)  

#Affichage du resultat / Displaying results
print(f"\nLe modele prédit que ce chiffre est : {predicted_class}")
print(f"Le chiffre reel est : {y_test[sample_index]}")
