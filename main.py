from keras.datasets import mnist
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
from keras import models
from network import Network
import numpy as np

skip = True

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images.reshape((train_images.shape[0], 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((test_images.shape[0], 28 * 28))
test_images = test_images.astype('float32') / 225

if not skip:
    num_classes = 10

    input_shape = (28, 28, 1)

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='sigmoid'))
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    batch_size = 128
    epochs = 15

    # network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    # network = models.load_model('model')

    # Calculate Test loss and Test Accuracy
    test_loss, test_acc = network.evaluate(test_images, test_labels)

m_network = Network()
m_network.add(8, 784)
m_network.add(10)

randomize = np.arange(len(train_images))
np.random.shuffle(randomize)

train_images_shuffled = train_images[randomize]
train_labels_shuffled = train_labels[randomize]

m_network.fit(train_images_shuffled, train_labels_shuffled, epochs=5, batch_size=128)
print("Test data evaluation")
m_network.evaluate(test_images, test_labels)
print("Train data evaluation")
m_network.evaluate(train_images, train_labels)
