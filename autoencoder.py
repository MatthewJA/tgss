"""Basic convolutional autoencoder.

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
image_width = 28  # Assume square.

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
    # Prime numbers are not great for widths, so drop the last column and row.
    images = images[:, :28, :28]
    train_images = images[:len(images) // 2]
    test_images = images[len(images) // 2:]
    return train_images, test_images

def get_model(conv=(3, 3)):
    img = keras.layers.Input((1, image_width, image_width))

    # Encoder
    # 1 x 28 x 28
    conv1 = keras.layers.Conv2D(16, conv, activation='relu', padding='same')(img)
    # 16 x 28 x 28
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)
    # 16 x 14 x 14
    conv2 = keras.layers.Conv2D(8, conv, activation='relu', padding='same')(pool1)
    # 8 x 14 x 14
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
    # 8 x 7 x 7
    conv3 = keras.layers.Conv2D(8, conv, activation='relu', padding='same')(pool2)
    # 8 x 7 x 7
    encoded = keras.layers.MaxPooling2D((2, 2))(conv3)
    # 8 x 4 x 4

    # Decoder
    # 8 x 4 x 4
    deconv1 = keras.layers.Conv2D(8, conv, activation='relu', padding='same')(encoded)
    # 8 x 4 x 4
    upsamp1 = keras.layers.UpSampling2D((2, 2))(deconv1)
    # 8 x 8 x 8
    deconv2 = keras.layers.Conv2D(8, conv, activation='relu', padding='same')(upsamp1)
    # 8 x 8 x 8
    upsamp2 = keras.layers.UpSampling2D((2, 2))(deconv2)
    # 8 x 16 x 16
    deconv3 = keras.layers.Conv2D(16, conv, activation='relu', padding='same')(upsamp2)
    # 16 x 16 x 16
    upsamp3 = keras.layers.UpSampling2D((2, 2))(deconv3)
    # 16 x 32 x 32
    decoded = keras.layers.Conv2D(1, (5, 5), acitvation='sigmoid', padding='same')(upsamp3)
    # 1 x 28 x 28

    autoencoder = keras.models.Model(img, decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # TODO(MatthewJA): Return an encoder and decoder too.
    return autoencoder

def train():
    autoencoder = get_model()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # Load and normalise images.
    train_images, test_images = load_tgss('/home/alger/myrtle1/tgss-cutouts/')
    autoencoder.fit(train_images, train_images,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(test_images, test_images))
    autoencoder.save('ae.h5')

if __name__ == '__main__':
    train()

