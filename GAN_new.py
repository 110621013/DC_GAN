from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #for CPU train
import matplotlib.pyplot as plt
import sys
import numpy as np


def load_mnist(batch_size, is_training=True):
    train_num, test_num = 60000, 10000
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((train_num, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((train_num)).astype(np.int32)

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((test_num, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((test_num)).astype(np.int32)

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch


def load_myself(batch_size, is_training=True):
    path = os.path.join('data', 'myself')
    train_num, test_num = 719, 359
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((train_num, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((train_num)).astype(np.int32)

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        fd = open(os.path.join(path, 'test-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((test_num, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 'test-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((test_num)).astype(np.int32)

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch


def load_npydata(batch_size, is_training=True):
    path = os.path.join('data', 'npydata')
    train_num, test_num = 700, 359
    if is_training:
        trainingdata = np.load(os.path.join(path, 'trainingdata.npy')) #(700, 538, 1000) ndarray
        trainX = trainingdata.reshape((train_num, length, width, 1)).astype(np.float32)

        traininglabel = np.load(os.path.join(path, 'traininglabel.npy')) #(700) ndarray
        trainY = traininglabel.reshape((train_num)).astype(np.int32)

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        testdata = np.load(os.path.join(path, 'testdata.npy'))
        teX = testdata.reshape((test_num, length, width, 1)).astype(np.float32)

        testlabel = np.load(os.path.join(path, 'testlabel.npy'))
        teY = testlabel.reshape((test_num)).astype(np.int32)

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True):
    if dataset == 'mnist':
        print('load_data:mnist')
        return load_mnist(batch_size, is_training)
    elif dataset == 'myself':
        print('load_data:myself')
        return load_myself(batch_size, is_training)
    elif dataset == 'npydata':
        print('load_data:npydata')
        return load_npydata(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)



class DCGAN():
    def __init__(self, dataset):
        # Input shape
        self.img_rows = length
        self.img_cols = width
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        if dataset == 'npydata':self.generator = self.build_generator2()
        else:self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    #7*2*2 7*2*2  ==>  269*2*1   2*5^3 *2*2
    def build_generator2(self):
        model = Sequential()
        model.add(Dense(128 * 269 * 250, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((269, 250, 128)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, save_interval, dataset):
        print('---------------------------Starting training')
        # Load the dataset
        X_train, _ , _ = load_data(dataset=dataset, batch_size=batch_size) #0~1 60000,28,28,1
        X_train = X_train * 2 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print("Epoch", epoch, epochs)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, dataset)

    def save_imgs(self, epoch, dataset):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('GAN_new_result/'+dataset+'/'+dataset+'_'+str(epoch)+'.png')
        plt.close()


if __name__ == '__main__':
    length, width = 538, 1000
    dcgan = DCGAN(dataset='npydata')
    dcgan.train(epochs=3000, batch_size=4, save_interval=100, dataset='npydata')

