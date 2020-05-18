# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.__version__

import glob
import os
from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras_preprocessing.image import img_to_array, load_img
import imageio
from keras.optimizers import Adam
import time
from skimage.io import imread
import cv2

"""## **The Generator Network**"""

# Hyperparameters required for generator network

residual_blocks = 16
momentum =0.8
input_shape = (64, 64, 3)

# Input layer to feed input to the network of a shape of (64,64,3)

input_layer = Input(shape=input_shape)
input_layer

# Adding the pre-residual block (2D convolution layer)
#filters: 64, kerel size: 9, strides: 1, Padding: same, Activation: relu

gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

# Method with the entire code for the residual block

def residual_block(x):
  filters = [64,64]
  kernel_size = 3
  strides = 1
  padding = 'same'
  momentum = 0.8
  activation = 'relu'

  res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
  res = Activation(activation=activation)(res)
  res = BatchNormalization(momentum=momentum)(res)

  res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
  res = BatchNormalization(momentum=momentum)(res)

  # Add res and x
  res = Add()([res, x])
  return res

# Now adding 16 residual blocks using residual_bloack function

res = residual_block(gen1)
for i in range(residual_blocks - 1):
  res = residual_block(res)

# Now wrap the entire code for the generator network inside a python function

def build_generator():
  # Hyper parameters
  residual_blocks = 16
  momentum = 0.8
  input_shape = (64, 64, 3)

  # input layer of the generator network
  input_layer = Input(shape=input_shape)

  # Add the preresidual block

  gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

  # Add 16 residual blocks

  res = residual_block(gen1)
  for i in range(residual_blocks - 1):
    res = residual_block(res)
  
  # Add post residual block

  gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
  gen2 = BatchNormalization(momentum=momentum)(gen2)

  # Take the sum of the output of pre seidual block and post residual block

  gen3 = Add()([gen2, gen1])

  # Add an Upsampling block

  gen4 = UpSampling2D(size=2)(gen3)
  gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
  gen4 = Activation('relu')(gen4)

  # Add another Upsampling block

  gen5 = UpSampling2D(size=2)(gen4)
  gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
  gen5 = Activation('relu')(gen5)

  # Output Convolution layer

  gen6 = Conv2D(filters=3, kernel_size = 9, strides=1, padding='same')(gen5)
  output = Activation('tanh')(gen6)

  # Keras model

  gen_model = Model(inputs=[input_layer], outputs=[output], name='generator')

  return gen_model

"""## **The Discriminator Network**"""

# Hyperparameters
# leakyrelu_alpha = 0.2
# momentum = 0.8
# input_shape = (256, 256, 3)

# input_layer = Input(shape=input_shape)


# Wrapping the entire code for the discriminator network inside a function

def build_discriminator():
  # create a discriminator using below hyperparameters
  leakyrelu_alpha = 0.2
  momentum = 0.8
  input_shape = (256, 256, 3)

  input_layer = Input(shape=input_shape)

  # Adding first Convolution block
  dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
  dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

  # Add another 7 convolution blocks with 
  # FIlters = 64, 128, 128, 256, 256, 512, 512
  # Kernel size = 3,3,3,3,3,3,3
  # Strides = 2,1,2,1,2,1,2
  # Padding: same for each convolution layer
  # Activation: LeakyRelu with alpha equal to 0.2 for each convolution

  # Add 2nd block

  dis2 = Conv2D(filters=64, kernel_size= 3, strides=2, padding='same')(dis1)
  dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
  dis2 = BatchNormalization(momentum=momentum)(dis2)

  # Add 3nd block

  dis3 = Conv2D(filters=64, kernel_size= 3, strides=1, padding='same')(dis2)
  dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
  dis3 = BatchNormalization(momentum=momentum)(dis3)

  # Add 4nd block

  dis4 = Conv2D(filters=64, kernel_size= 3, strides=2, padding='same')(dis3)
  dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
  dis4 = BatchNormalization(momentum=momentum)(dis4)

  # Add 5nd block

  dis5 = Conv2D(filters=64, kernel_size= 3, strides=1, padding='same')(dis4)
  dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
  dis5 = BatchNormalization(momentum=momentum)(dis5)

  # Add 6nd block

  dis6 = Conv2D(filters=64, kernel_size= 3, strides=2, padding='same')(dis5)
  dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
  dis6 = BatchNormalization(momentum=momentum)(dis6)

  # Add 7nd block

  dis7 = Conv2D(filters=64, kernel_size= 3, strides=1, padding='same')(dis6)
  dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
  dis7 = BatchNormalization(momentum=momentum)(dis7)

  # Add 8nd block

  dis8 = Conv2D(filters=64, kernel_size= 3, strides=2, padding='same')(dis7)
  dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
  dis8 = BatchNormalization(momentum=momentum)(dis8)

  # Adding a Dense layer

  dis9 = Dense(units=1024)(dis8)
  dis9 = LeakyReLU(alpha=leakyrelu_alpha)(dis9)

  # Adding a Dense Layer for Classification

  output = Dense(units=1, activation='sigmoid')(dis9)

  # Creating a kerass model for inputs and outputs

  dis_model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
  return dis_model

"""## **VGG19 Network**

We will use the pretrained VGG19 network. The purpose of the VGG19 network
 is to extract feature maps of the generated and the real images.
"""

# Now wrapping the entire code in a function

def build_vgg():
  # Build the vgg network to extract image features

  input_shape = (256, 256, 3)

  # Loading a pretrained VGG model on imagenet dataset
  vgg = VGG19(weights='imagenet')
  vgg.outputs = [vgg.layers[9].output]

  input_layer = Input(shape=input_shape) 

  # Extracting features from vgg

  features = vgg(input_layer)

  # Create a keras model
  model = Model(inputs=[input_layer], outputs=[features])

  return model

"""## **The Adversarial Network**

The adversarial network is a combined network that uses the generator, thr discriminator, and the VGG19.
"""

# Wrap the entire code for the advesarial model

def build_adversarial_model(generator, discriminator, vgg):
  # Start by creating an input layer for the network

  input_low_resolution = Input(shape=(64, 64, 3))

  # Next generate fake High Resolution images from generator network

  fake_hr_images = generator(input_low_resolution)

  # Next extract the features of the fake images using VGG19

  fake_features = vgg(fake_hr_images)

  # Make the discriminator network non-trainable in the adversarial network as we are traing the generatot

  discriminator.trainable = False

  # Next, pass the fake images to the discriminator

  output = discriminator(fake_hr_images)

  # Finally create a keras model, which will be our adversarial network

  model = Model(inputs=[input_low_resolution], outputs=[output, fake_features])

  for layer in model.layers:
    print(layer.name, layer.trainable)

  print(model.summary())
  return model

"""####**Training the SRGAN**

Training the SRGAN is a two step process where we first train the discriminator and then train the adversarial network which eventually trains the generator network.
"""

# Start by defining the hyperparameters required for the training

data_dir = "/content/drive/My Drive/traindata"
epochs = 20000
batch_size = 1

# Shape of low-resolution and high-resolution images

low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

# Next, define the training optimizer. For all the networks we use Adam optimizer with learning ratet equal to 0.0002 and beta_1 equal to 0.5

common_optimizer = Adam(0.0002, 0.5)

"""### **Building and Compiling the Networks**"""

# Build and compile vgg
vgg = build_vgg()
vgg.trainable = False
vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

# Build and compile Discriminator

discriminator = build_discriminator()
discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

# Building Generator Network

generator = build_generator()

# Create an adversarial mode. Start by creating two input layers

input_high_resolution = Input(shape = high_resolution_shape)
input_low_resolution = Input(shape = low_resolution_shape)

# Use the generator network to symbolycally generate high resolution images from low resolution image

generated_high_resolution_images = generator(input_low_resolution)

# Use vgg to extract feeature maps

features = vgg(generated_high_resolution_images)

# Make the discriminator network non trainable

discriminator.trainable = False

# Use discriminator to get probabilities of generated fake images

probs = discriminator(generated_high_resolution_images)

# Finally create and compile the adversarial network

adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer)
# 1e = 0.001

# Adding tensorboard to visualize the training losses and to visualize the network graphs

tensorboard = TensorBoard(log_dir="/content/drive/My Drive/srgan".format(time.time()))
tensorboard.set_model(generator)
tensorboard.set_model(discriminator)

# Creating a loop that should run for the specified number of epochs

low_resolution_shape_cv = (64, 64)
high_resolution_shape_cv = (256, 256)

def sample_images(data_dir, batch_size, high_resolution_shape_cv, low_resolution_shape_cv):

  all_images = os.listdir(data_dir)

  images_batch = np.random.choice(all_images, size=batch_size)

  low_resolution_images = []
  high_resolution_images = []


  for img in images_batch:
    n= cv2.imread(os.path.join(data_dir,img))

    img1_high_resolution = cv2.resize(n, high_resolution_shape_cv)
    img1_low_resolution = cv2.resize(n, low_resolution_shape_cv)

    if np.random.random() <0.5:
      img1_high_resolution = np.fliplr(img1_high_resolution)
      img1_low_resolution = np.fliplr(img1_low_resolution)
        
    high_resolution_images.append(img1_high_resolution)
    low_resolution_images.append(img1_low_resolution)
  return np.array(high_resolution_images), np.array(low_resolution_images)

for epoch in range(epochs):
  print("Epoch:{}".format(epoch))
  high_resolution_images,low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                               low_resolution_shape_cv=low_resolution_shape_cv, 
                                                               high_resolution_shape_cv = high_resolution_shape_cv)

high_resolution_images = high_resolution_images / 127.5 - 1
low_resolution_images = low_resolution_images / 127.5 - 1

# Training the discriminator network
# Generate fake high resolution images

generated_high_resolution_images = generator.predict(low_resolution_images)

# Create a batch of real labels and fake labels

real_labels = np.ones((batch_size, 16, 16, 1))
fake_labels = np.zeros((batch_size, 16, 16, 1))

# Train the discriminator network on real images and real labels:

d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)

# Train the discriminator on generated images and fake labels:

d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)

# Finally calculate the total discriminator loss

d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

# Now we added the code to train the discriminator network, next we add the code to train the adversarial model, which trains the generator
# Again sample a batch of high resolution and low resolution images and normalize them

high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir,
                                                              batch_size=batch_size,low_resolution_shape_cv=low_resolution_shape_cv,
                                                              high_resolution_shape_cv=high_resolution_shape_cv)
# Normalize images

high_resolution_images = high_resolution_images / 127.5 - 1
low_resolution_images = low_resolution_images / 127.5 - 1 

# Use the VGG19 to get feature maps of real high resolution images

image_features = vgg.predict(high_resolution_images)

# Finally, train the adversarial model and provide it with appropriate inputs

g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],[real_labels, image_features])

# After completion of each epoch, write the losses to tensorboard to visualize



#After every 100 epochs, generate high_resolutoin fake images using the generatornetwork and save them to visualize


generator.save("/content/drive/My Drive/srgan/gmodel2.h5")
discriminator.save("/content/drive/My Drive/srgan/dmodel2.h5")

def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save images in a single figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(low_resolution_image)
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image)
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(path)



# Build the generator network
generator = build_generator()

# Load models
generator.load_weights("/content/drive/My Drive/srgan/gmodel2.h5")

# Get 10 random images
high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=1,
                                                              low_resolution_shape_cv=low_resolution_shape_cv,
                                                              high_resolution_shape_cv=high_resolution_shape_cv)
# Normalize images
high_resolution_images = high_resolution_images / 127.5 - 1.
low_resolution_images = low_resolution_images / 127.5 - 1.

generated_images = generator.predict_on_batch(low_resolution_images)

for index, img in enumerate(generated_images):
    save_images(low_resolution_images[index], high_resolution_images[index], img,
                path="/content/drive/My Drive/srgan/results/gen_{}".format(index))