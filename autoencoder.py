"""Basic convolutional autoencoder.

Based on Keras examples at
https://blog.keras.io/building-autoencoders-in-keras.html.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import argparse
import os

import astropy.io.fits
import keras.layers
import keras.models
import numpy

import tgss

# Dimensionality of encoding.
hidden_size = 32

# Input image width, px.
image_width = 28  # Assume square.

def load_tgss(tgss, tgss_path):
    images = []
    for filename in os.listdir(tgss_path):
        if not filename.lower().endswith('fits'):
            continue
        # Eliminate objects that are compact.
        name = filename.split('_')[0]
        if tgss.is_compact(name):
            continue
        with astropy.io.fits.open(os.path.join(tgss_path, filename)) as fits:
            images.append(fits[0].data.copy())
    images = numpy.array(images)
    images = numpy.nan_to_num(images)
    images -= images.min()
    images /= images.max()
    # Prime numbers are not great for widths, so drop the last column and row.
    images = images[:, :28, :28]
    images = images.reshape((-1, 1, 28, 28))
    train_images = images[:len(images) // 2]
    test_images = images[len(images) // 2:]
    return train_images, test_images

def get_model(conv=(3, 3)):
    img = keras.layers.Input((1, image_width, image_width))

    # 1 x 28 x 28
    conv1 = keras.layers.Conv2D(16, conv, activation='relu', padding='same', data_format='channels_first')(img)
    # 16 x 28 x 28
    pool1 = keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    # 16 x 14 x 14
    conv2 = keras.layers.Conv2D(8, conv, activation='relu', padding='same', data_format='channels_first')(pool1)
    # 8 x 14 x 14
    pool2 = keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    # 8 x 7 x 7
    conv3 = keras.layers.Conv2D(8, conv, activation='relu', padding='same', data_format='channels_first')(pool2)
    # 8 x 7 x 7
    encoded = keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(conv3)
    # 8 x 3 x 3

    # Decoder
    # 8 x 3 x 3
    deconv1 = keras.layers.Conv2D(8, conv, activation='relu', padding='same', data_format='channels_first')(encoded)
    # 8 x 3 x 3
    upsamp1 = keras.layers.UpSampling2D((3, 3), data_format='channels_first')(deconv1)
    # 8 x 9 x 9
    deconv2 = keras.layers.Conv2D(8, conv, activation='relu', padding='same', data_format='channels_first')(upsamp1)
    # 8 x 9 x 9
    upsamp2 = keras.layers.UpSampling2D((2, 2), data_format='channels_first')(deconv2)
    # 8 x 18 x 18
    deconv3 = keras.layers.Conv2D(16, conv, activation='relu', padding='same', data_format='channels_first')(upsamp2)
    # 16 x 18 x 18
    upsamp3 = keras.layers.UpSampling2D((2, 2), data_format='channels_first')(deconv3)
    # 16 x 36 x 36
    deconv4 = keras.layers.Conv2D(16, (7, 7), activation='relu', padding='valid', data_format='channels_first')(upsamp3)
    # 16 x 32 x 32
    decoded = keras.layers.Conv2D(1, conv, activation='relu', padding='valid', data_format='channels_first')(deconv4)
    # 1 x 28 x 28

    autoencoder = keras.models.Model(img, decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # TODO(MatthewJA): Return an encoder and decoder too.
    return autoencoder

def train(tgss, output):
    autoencoder = get_model()

    # Load and normalise images.
    train_images, test_images = load_tgss(tgss, '/home/alger/myrtle1/tgss-cutouts/')
    autoencoder.fit(train_images, train_images,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(test_images, test_images))
    autoencoder.save(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='ae.h5')
    args = parser.parse_args()
    t = tgss.TGSS(
        '/home/alger/myrtle1/tgss/',
        '/home/alger/myrtle1/tgss/TGSSADR1_7sigma_catalog.tsv',
        '/home/alger/myrtle1/tgss/grid_layout.rdb')
    train(t, args.output)
