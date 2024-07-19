# Image-Generation-using-GAN
Generative Adversarial Networks consist of two neural networks: the generator and the discriminator. The generator creates fake data, while the discriminator evaluates the authenticity of the data (real or fake). The two networks are trained simultaneously, with the generator trying to produce realistic data to fool the discriminator, and the discriminator trying to correctly classify the data.

GENERATOR,
The Generator class is defined to create fake images from random noise (latent space). Key components include:
  Initialization: The constructor initializes the model architecture using fully connected layers.
  Block Function: A helper function that creates a sequence of layers, including linear transformations, batch normalization, and activation functions (LeakyReLU).
  Forward Method: This method takes a latent vector z and transforms it into an image.

DISCRIMINATOR,
  The Discriminator class is responsible for distinguishing between real and fake images. Its components include:
  Initialization: Similar to the generator, it initializes a model with fully connected layers.
  Forward Method: This method flattens the input image and passes it through the model to output a probability indicating whether the image is real or fake.

The MNIST dataset is loaded and transformed. The transformations include resizing, normalization, and conversion to tensors. A DataLoader is created to facilitate batching and shuffling of the dataset.

While we are training the GAN,
  Epoch Loop: Iterates over the specified number of epochs.
  Batch Loop: Iterates over batches of images from the DataLoader.
  Adversarial Ground Truths: Creates labels for real (1) and fake (0) images.
  Generator Training: The generator is trained to produce images that can fool the discriminator. The loss is computed based on the discriminator's output for     generated images.
  Discriminator Training: The discriminator is trained on both real and generated images. The average loss from both real and fake images is calculated.

Finally, The save_image function is defined to save generated images during training. It uses the save_image function from torchvision.utils to create image files.
