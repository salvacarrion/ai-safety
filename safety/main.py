import os
import json
import datetime
import math
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn import metrics


def show_image(img, shape=(28, 28)):
    if shape:
        plt.imshow(img.reshape(shape))
    else:
        plt.imshow(img)
    plt.show()


def get_data_mnist(target_class=None):
    num_classes = 10
    input_shape = (28*28,)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28)
    x_train = x_train.reshape(len(x_train), *input_shape)
    x_test = x_test.reshape(len(x_test), *input_shape)

    # # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Binary classifier
    if target_class:
        num_classes = 2
        y_train = y_train == target_class
        y_test = y_test == target_class

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def get_model(train_size, input_shape, num_classes):
    m_layers = []

    # Input layer (fixed)
    m_layers.append(keras.Input(shape=input_shape))

    # Add random layers
    m_layers += get_rnd_layers(train_size=train_size)

    # Output layer (fixed)
    m_layers.append(layers.Dense(num_classes, kernel_initializer="normal", activation="softmax"))

    # Build model
    model = keras.Sequential(m_layers)
    return model


def get_rnd_layers(train_size, min_layers=1, max_layers=5):
    def get_rnd_neurons():
        ls = int(np.log(train_size) * 26)  # Heuristic constant: [22, 28] => 26
        ls = max(ls, 120)  # Set a minimum number of neurons
        return math.floor(random.gauss(ls, ls // 16))

    # Generate random layers
    rnd_layers = []
    num_layers = random.randint(min_layers, max_layers)  # Arbitrary
    for i in range(num_layers):
        rnd_neurons = get_rnd_neurons()
        rnd_layers.append(layers.Dense(rnd_neurons, kernel_initializer="random_uniform", activation="relu"))

    return rnd_layers


def train_model(model, x_train, y_train, batch_size, epochs):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.0, shuffle=True)


def evaluate_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test, verbose=0)

def main():
    # Constants
    NUM_RUNS = 5
    NUM_HERDS = 3
    BATCH_SIZE = 128
    MAX_EPOCHS = 3
    TARGET_CLASS = 1  # To binarize the problem

    # Stats
    accuracies = []

    # Get data
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_data_mnist(target_class=TARGET_CLASS)

    # Repeat process n times
    for i in range(NUM_RUNS):
        h_scores = []
        h_predictions = []

        for j in range(NUM_HERDS):
            # Generate model
            model = get_model(len(x_train), input_shape, num_classes)
            model.summary()

            # Train model
            train_model(model, x_train, y_train, BATCH_SIZE, MAX_EPOCHS)

            # Store predictions
            m_predictions = model.predict(x_test)
            h_predictions.append(m_predictions)

            # Evaluate
            m_scores = evaluate_model(model, x_test, y_test)
            h_scores.append(m_scores)

        # Majority vote
        h_predictions = np.array(h_predictions)
        avg_predictions = np.mean(h_predictions, axis=0)

        # Get max class prob
        max_class_y_true = np.argmax(y_test, axis=1)
        max_class_y_pred = np.argmax(avg_predictions, axis=1)

        # Compute metrics
        accuracy = metrics.accuracy_score(max_class_y_true, max_class_y_pred)
        accuracies.append(accuracy)

    # Collect results
    results = {"accuracies": accuracies,
               "mean_acc": np.mean(accuracies),
               "std_acc": np.std(accuracies),
               "n_herds": NUM_HERDS,
               "n_runs": NUM_RUNS,
               "batch_size": BATCH_SIZE,
               "max_epochs": MAX_EPOCHS,
               "num_classes": num_classes,
               "target_class": TARGET_CLASS
               }

    # Save data
    filename = f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')}.json"
    with open(os.path.join("data", filename), 'w') as f:
        json.dump(results, f)
    print("Results saved!")

    # Print summary
    print("")
    print("****************************************************")
    print("**** SUMMARY ***************************************")
    print("****************************************************")
    for k, v in results.items():
        if k != "accuracies":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
    print("Done!")
