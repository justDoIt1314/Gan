from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
from tensorflow.python.keras import layers,losses,models,metrics
import numpy as np
import datetime
import os
import cv2
import matplotlib.pyplot as plt
import time
from IPython import display
batch_size=128
noise_dim = 128
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train-127.5)/127.5
x_train = x_train.reshape(-1,28,28,1).astype('float32')
x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)
tf.keras.Model
def make_generator_model(inputs):
    # model = tf.keras.Sequential()
    # model.add(layers.Dense(7*7*256,input_shape=inputs))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Reshape(target_shape=(7,7,256)))
    # assert model.output_shape == (None,7,7,256)

    # model.add(layers.Conv2DTranspose(128,(5,5),padding='same'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    # assert model.output_shape == (None,7,7,128)

    # model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    # assert model.output_shape == (None,14,14,64)
    # model.add(layers.Conv2DTranspose(1,(3,3),strides=(2,2),padding='same',activation='tanh'))
    # assert model.output_shape == (None,28,28,1)

    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=inputs))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
      
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)  
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    # model = tf.keras.Sequential()
    # model.add(layers.Dense(1024,input_shape=inputs))
    # model.add(layers.Activation('tanh'))
    # model.add(layers.Dense(128 * 7 * 7))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation('tanh'))
    # model.add(layers.Reshape((7, 7, 128), input_shape=(7 * 7 * 128,)))
    # model.add(layers.UpSampling2D(size=(2, 2)))
    # model.add(layers.Conv2D(64, (5, 5), padding='same'))
    # model.add(layers.Activation('tanh'))
    # model.add(layers.UpSampling2D(size=(2, 2)))
    # model.add(layers.Conv2D(1, (5, 5), padding='same'))
    # model.add(layers.Activation('tanh'))

    return model

generator_model = make_generator_model((noise_dim,))

noise = tf.random.normal(shape=(1,noise_dim))
noise_2 = tf.random.normal(shape=(64,noise_dim))
gene_image = generator_model(noise)
image = gene_image[0,:,:,0]

def make_discriminator_model(inputs):
    # model = tf.keras.Sequential([
    #     layers.Conv2D(64,(3,3),input_shape=inputs),
    #     layers.LeakyReLU(),
    #     layers.Conv2D(128,(3,3)),
    #     layers.LeakyReLU(),
    #     layers.Dropout(0.2),
    #     layers.Flatten(),
    #     layers.Dense(64,activation='sigmoid'),
    #     layers.Dropout(0.2),
    #     layers.Dense(1)
    # ])

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), activation='tanh', padding='same', 
                                     input_shape=inputs))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
      
    model.add(layers.Conv2D(128, (5, 5), activation='tanh', padding='same'))
    
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='tanh'))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model

discriminator_model = make_discriminator_model(inputs=(28,28,1))

decision = discriminator_model(gene_image)
print(decision)

def generator_loss(fake_output):
    return tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output),fake_output)

def discriminator_loss(fake_output,real_output):
    # real_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output),real_output)
    # fake_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output),fake_output)

    ones_zeros = tf.concat((tf.ones_like(real_output),tf.zeros_like(fake_output)),0)
    real_fake = tf.concat((real_output,fake_output),0)
    dis_loss = tf.losses.BinaryCrossentropy()(ones_zeros,real_fake)
    return dis_loss

gen_optimizer = tf.keras.optimizers.SGD(0.001,name='gen')
dis_optimizer = tf.keras.optimizers.SGD(0.001,name='dis')
train_gen_loss = tf.keras.metrics.Mean('gen_loss')
train_dis_loss = tf.keras.metrics.Mean('dis_loss')
@tf.function
def train_step(images):
    noise_1 = tf.random.normal(shape=[batch_size,noise_dim],seed=2999)
    
    #noise = tf.random.normal(shape=(64,100))
    with tf.GradientTape() as gen_tap,tf.GradientTape() as dis_tap:
        gene_image = generator_model(noise_1,training=True)
        # cv2.imshow('dsf',gene_image.numpy()[0,:,:,0])
        # cv2.waitKey(10)
        fake_output = discriminator_model(gene_image,training=True)
        real_output = discriminator_model(images,training=True)
        gene_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(fake_output,real_output)
      
        gen_gradient = gen_tap.gradient(gene_loss,generator_model.trainable_variables)
        dis_gradient = dis_tap.gradient(dis_loss,discriminator_model.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradient,generator_model.trainable_variables))
        dis_optimizer.apply_gradients(zip(dis_gradient,discriminator_model.trainable_variables))

        train_dis_loss(dis_loss)
        train_gen_loss(gene_loss)

checkPath = 'checkpoint'
if not os.path.exists('checkpoint'):
    os.mkdir('checkpoint')

checkpoint = tf.train.Checkpoint(generator_model=generator_model,gen_optimizer=gen_optimizer,
discriminator_model=discriminator_model,dis_optimizer=dis_optimizer)

num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

noise_2 = tf.random.normal(shape=[batch_size,noise_dim],seed=2999)
def train(dataset,epoches):
    step_count=0
    for epoch in range(epoches):
        start = time.time()
        for batch_images in dataset:
            train_step(batch_images)
            with train_summary_writer.as_default():
                tf.summary.scalar('gen_loss', train_gen_loss.result(), step=step_count)
                tf.summary.scalar('dis_loss',train_dis_loss.result(),step=step_count)
            gene_image = generator_model(noise_2,training=False)
            step_count = step_count + 1
            cv2.imshow('dsf',gene_image.numpy()[0,:,:,0])
            cv2.waitKey(10)

        #display.clear_output(wait=True)
        generator_and_save_image(generator_model,epoch+1,seed)
        if (epoch+1)%20==0:
            checkpoint.save(os.path.join(checkPath,'ckpt'))            
        print('time for epoch is {} second'.format(time.time()-start))




def generator_and_save_image(model,epoch,test_input):
    prediction = model(test_input,training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(prediction.shape[0]):
        plt.subplot(4,4,i+1)
        #plt.imshow(,)
        plt.imshow(prediction[i,:,:,0]*127.5+127.5,cmap='gray')
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()


train(x_train,500)
