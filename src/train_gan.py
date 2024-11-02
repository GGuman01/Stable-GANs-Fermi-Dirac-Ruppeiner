import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import scipy.constants as const

# Global parameters for Fermi-Dirac distribution
TEMPERATURE = 1.0  # Adjustable for paper replication
CHEMICAL_POTENTIAL = 0.5  # Threshold in GAN context
BOLTZMANN_CONSTANT = const.Boltzmann  # kB

# Fermi-Dirac distribution for state occupancy
def fermi_dirac_distribution(energy, temperature=TEMPERATURE, mu=CHEMICAL_POTENTIAL):
    return 1 / (np.exp((energy - mu) / (BOLTZMANN_CONSTANT * temperature)) + 1)

# Generator model definition
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(28 * 28, activation='sigmoid'))  # Output for MNIST
    model.add(layers.Reshape((28, 28)))
    return model

# Discriminator model definition
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    return model

# Function to combine generator and discriminator into GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([generator, discriminator])
    return model

# Simplified Ruppeiner curvature calculation (placeholder)
def calculate_ruppeiner_curvature(loss_values):
    if len(loss_values) < 2:
        return np.nan  # Insufficient data
    gradient = np.gradient(loss_values)
    second_derivative = np.gradient(gradient)
    curvature = -0.5 * second_derivative.mean()
    return curvature

# GAN training function
def train_gan(generator, discriminator, gan, epochs, batch_size, data, latent_dim):
    half_batch = batch_size // 2
    loss_values = []
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_images = data[idx]
        real_labels = np.ones((half_batch, 1))
        
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
        # Compute average Discriminator loss
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))  # Trick the discriminator
        g_loss = gan.train_on_batch(noise, valid_labels)
        
        # Record Generator loss for analysis
        loss_values.append(g_loss)
        
        # Periodic status updates with curvature analysis
        if epoch % 10 == 0:
            curvature = calculate_ruppeiner_curvature(loss_values)
            print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
            print(f"Ruppeiner curvature: {curvature:.4f}")
            if curvature > 0:
                print("Training is stable.")
            else:
                print("Potential instability detected.")

# Prepare MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 255.0  # Normalize input data
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

# Set GAN parameters
latent_dim = 100

# Build and compile models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# Execute training
train_gan(generator, discriminator, gan, epochs=150, batch_size=64, data=x_train, latent_dim=latent_dim)
