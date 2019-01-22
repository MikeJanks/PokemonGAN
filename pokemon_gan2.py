import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sqrt


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Reshape, UpSampling2D, Activation, ZeroPadding2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

random_dim = 100

def resize(img, size):
    dim = (size, size)
    resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resize


def load_pokemon_data():
    imgs = []
    files = os.listdir(".\\kaggle-one-shot-pokemon\\pokemon-b")
    for x in tqdm(files):
        img = cv2.imread(".\\kaggle-one-shot-pokemon\\pokemon-b\\"+str(x), cv2.IMREAD_UNCHANGED)
        img = resize(img, 32)
        imgs.append(img)
    
    print(len(imgs))
    
    imgs = np.array(imgs)
    # imgs = imgs.reshape(len(files), 3136)
    return imgs


def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)
    # return Adam(lr=0.00002)


def get_generator():
    generator = Sequential()

    generator.add(Dense(4*4*512, kernel_initializer=initializers.RandomNormal(stddev=0.02), input_dim=random_dim))
    generator.add(Reshape((4, 4, 512)))
    generator.add(BatchNormalization())
    # generator.add(LeakyReLU(alpha=0.2))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(256, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, padding="same"))
    generator.add(BatchNormalization())
    # generator.add(LeakyReLU(alpha=0.2))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(128, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, padding="same"))
    generator.add(BatchNormalization())
    # generator.add(LeakyReLU(alpha=0.2))
    generator.add(Activation('relu'))

    # generator.add(Conv2DTranspose(64, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, padding="same"))
    # generator.add(BatchNormalization(momentum=0.5))
    # generator.add(LeakyReLU(alpha=0.2))
    # generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(3, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, padding="same"))
    generator.add(Activation("tanh"))

    generator.summary()

    return generator


def get_discriminator():
    discriminator = Sequential()

    discriminator.add(Conv2D(64, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, input_shape=(32,32,3), padding="same"))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    # discriminator.add(Dropout(0.25))

    # discriminator.add(Conv2D(128, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, input_shape=(32,32,3), padding="same"))
    discriminator.add(Conv2D(128, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, padding="same"))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    # discriminator.add(Dropout(0.25))

    discriminator.add(Conv2D(256, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, padding="same"))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    # discriminator.add(Dropout(0.25))

    # discriminator.add(Conv2D(512, kernel_initializer=initializers.RandomNormal(stddev=0.02), kernel_size=5, strides=2, padding="same"))
    # discriminator.add(BatchNormalization(momentum=0.5))
    # discriminator.add(LeakyReLU(alpha=0.2))
    # discriminator.add(Dropout(0.25))
    
    discriminator.add(Flatten())
    discriminator.add(Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.02), activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5))

    discriminator.summary()
    # input('paused...')

    return discriminator


def get_gan_network(discriminator, random_dim, generator):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5))
    return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 32, 32, 3)

    plt.figure(1,figsize=figsize)
    plt.clf()
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(cv2.cvtColor(generated_images[i], cv2.COLOR_BGR2RGB), interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('.\\gen2_pokemon\\gan_generated_image_epoch_%d.png' % epoch)

def plot_graph_disc_and_gan(discriminate_loss_list, generate_loss_list):
    plt.figure(2)
    plt.clf()
    plt.plot(generate_loss_list, c='g')
    plt.plot(discriminate_loss_list, c='r')
    plt.savefig('.\\gen2_pokemon\\Generator_Discriminator_Graph.png')

def plot_graph_diff(diff_list):
    plt.figure(3)
    plt.clf()
    plt.plot(diff_list, c='b')
    plt.savefig('.\\gen2_pokemon\\Diff_Graph.png')
    

def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train = load_pokemon_data()
    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] / batch_size

    # Build our GAN netowrk
    # adam = get_optimizer()
    generator = get_generator()
    discriminator = get_discriminator()
    gan = get_gan_network(discriminator, random_dim, generator)

    modify = True
    discriminate_loss_list  = []
    generate_loss_list      = []
    diff_list               = []
    
    # Labels for generated and real data
    y_dis_real = np.random.normal(0, .1, size=batch_size)
    y_dis_fake = np.random.normal(.9, 1, size=batch_size)
    # y_dis_real = np.ones(batch_size)
    # y_dis_fake = np.zeros(batch_size)
    y_gen = np.random.normal(0, .1, size=batch_size*2)
    # y_gen = np.zeros(batch_size*2)

    e = 0
    while True:
        e += 1
        disc_loss   = 0
        gan_loss    = 0
        diff_temp   = 0
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(int(batch_count))):

            # Train Discriminator x times per iteration
            disc_train_times = 1
            disc_loss_temp = 0
            for _ in range(disc_train_times):
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])

                # Generate fake pokemon images
                generated_images = generator.predict(noise)
                # X = np.concatenate([image_batch, generated_images])


                # Train discriminator
                discriminator.trainable = True
                disc_loss_temp += discriminator.train_on_batch(image_batch, y_dis_real)
                disc_loss_temp += discriminator.train_on_batch(generated_images, y_dis_fake)

            disc_loss = disc_loss_temp/(disc_train_times*2)
            discriminate_loss_list.append(disc_loss)
            

            # Train GAN x times per iteration
            gan_train_times = 1
            gan_loss_temp = 0
            for _ in range(gan_train_times):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size*2, random_dim])


                # Train discriminator
                discriminator.trainable = False
                gan_loss_temp += gan.train_on_batch(noise, y_gen)

            gan_loss = gan_loss_temp/gan_train_times
            generate_loss_list.append(gan_loss)
            plot_graph_disc_and_gan(discriminate_loss_list, generate_loss_list)

            diff_temp += sqrt((disc_loss - gan_loss)**2)

        # Take the difference between the the losses and plot them
        diff = diff_temp / batch_count
        diff_list.append(diff)
        plot_graph_diff(diff_list)


            # if len(diff_list) > 10 and modify:
            #     modify = False
            #     diff_list = diff_list[50:]
            #     discriminate_loss_list = discriminate_loss_list[50:]
            #     generate_loss_list = generate_loss_list[50:]


        if e == 1 or e % 10 == 0:
            generator.save(".\\models2\\generator_epoch_"+str(e)+".model")
            discriminator.save(".\\models2\\discriminator_epoch_"+str(e)+".model")
            gan.save(".\\models2\\gan_epoch_"+str(e)+".model")
            plot_generated_images(e, generator)
        
        # disc_loss /= batch_count
        # gan_loss /= batch_count

        # discriminate_loss_list.append(disc_loss)
        # generate_loss_list.append(gan_loss)

        # plot_graph_disc_and_gan(discriminate_loss_list, generate_loss_list)
        # plot_graph_diff(diff_list)

        # print('Generator Loss:    ', gan_loss)
        # print('Discriminator Loss:', disc_loss)

if __name__ == '__main__':
    train(20000,32)