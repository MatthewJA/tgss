"""Basic autoencoder.

Based on Keras examples at
https://blog.keras.io/building-autoencoders-in-keras.html.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import os

import astropy.io.fits
import keras.layers
import keras.models
import numpy

# Dimensionality of encoding.
hidden_size = 32

# Input image width, px.
image_width = 29  # Assume square.

def load_tgss(tgss_path):
    images = []
    for filename in os.listdir(tgss_path):
        if not filename.lower().endswith('fits'):
            continue
        with astropy.io.fits.open(os.path.join(tgss_path, filename)) as fits:
            images.append(fits[0].data.copy())
    images = numpy.array(images)
    images = numpy.nan_to_num(images)
    images -= images.min()
    images /= images.max()
    images = images.reshape((-1, image_width ** 2))
    train_images = images[:len(images) // 2]
    test_images = images[len(images) // 2:]
    return train_images, test_images

def get_model():
    input_image = keras.layers.Input((image_width ** 2,))
    encoded = keras.layers.Dense(hidden_size, activation='relu')(input_image)
    output_image = keras.layers.Dense(image_width ** 2, activation='sigmoid')(encoded)

    input_encoded = keras.layers.Input((hidden_size,))

    autoencoder = keras.models.Model(input_image, output_image)
    encoder = keras.models.Model(input_image, encoded)
    decoder = keras.models.Model(
        input_encoded, autoencoder.layers[-1](input_encoded))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder, decoder

def train():
    autoencoder, *_ = get_model()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # Load and normalise images.
    train_images, test_images = load_tgss('/home/alger/myrtle1/tgss-cutouts/')
    autoencoder.fit(train_images, train_images,
                    epochs=500,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(test_images, test_images))
    autoencoder.save('ae.h5')

if __name__ == '__main__':
    train()

