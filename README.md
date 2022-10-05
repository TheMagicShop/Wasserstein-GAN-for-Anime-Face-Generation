# Wasserstein GAN for Anime Face Generation

Wasserstein GAN is an improvement upon the traditional Vanilla GAN, it comes to resolve some problems like mode collapse and ameliorate stability, mainly through the Wasserstein (or EMD) loss function, which has a solid and heavy mathematical background [2], but gone through many simplifications to become as easy as
`-mean(y_true * y_pred)` in addition to some weights clipping `clip(weights, -value, value)`.
<br/>
<br/>
In this project we are going to train a Wasserstein GAN in order to generate new Anime Faces, but the elaborated OOP code is flexible and can be used on any other image data to train it on your own dataset as explained below, all the code is well commentated and self-explanatory.
<br/>
<br/>
Anime Faces dataset can be easily downloaded from kaggle, refer to *demonstration_on_Anime_Faces.ipynb*
<br/>
<br/>
the code is written in Python using TensorFlow 2.
<br/>
<br/>

# Results

after having trained the GAN with `image_shape=(64, 64, 3)`, `noise_dim=128`, `learning_rate=5e-5`, `n_critic=10`, `clip_value=0.01` and a `batch_size=64`, for `train_steps=20000` (all this can be found on *demonstration_on_Anime_Faces.ipynb*), we achieved this quality of image generatrion.

after n_steps:

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/ComingSoon.jpg)


after m_steps:

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/ComingSoon.jpg)


we took the first and last components of a noise vector and make them vary along $[-1, +1]$ and this is the resulting grid of images:

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/ComingSoon.jpg)


we took two random noise vectors and average them up using different weights $\alpha$, `new_vector = alpha*v1 + (1-alpha)*v2`, these are the two images generated from the former noise vectors accompanied with the grid of the new images with $\alpha$ varying from $0$ to $1$:

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/ComingSoon.jpg)
<br/>
![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/ComingSoon.jpg)

this is a gif of a random image synthesized using a random vector being updated after each training step.

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/ComingSoon.jpg)

and finally this is the loss progress during training.

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/ComingSoon.jpg)
<br/>
<br/>

# Architecture

We begin our GAN with building the generator which starts with a 1-dimensional input layer of size `noise_dim`, then a Dense layer that projects the input into a higher dimensionality space, in such a way that its output is able to be reshaped into a small and deep tensor of feature maps of the form $(x,y,n)$ where $(x,y)$ ~ $(4,4)$ and $n$ is $512$, and then a stack of blocks (BatchNormalization - ReLU - Conv2DTranspose), where each convolutional layer decrease the depth (feature maps) and upsample the size of the image, we employ as much of these stacks until we reach the original size, namely $(64, 64, 32)$, then two other blocks but this time with no upsampling as we've already reached the orginal size, but only to reach the number of channels $(64, 64, 3)$, and finally we add a sigmoid activation layer to squash the tensor values inside $[0,1]$ to become a proper RGB image.

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/figures/generator_architecture.png)

On the other hand there is the discriminator which mainly is the reverse operation of the generation, it downsamples the image until we reach the shape $(x,y,n)$ mentioned earlier, and we follow it by a Flatten layer and a Dense layer (with linear activation) of one single neuron to discriminate the realness of the feed image, but this time the blocks consist of (LeakyReLU - Conv2D), the output of the Dense layer are either positive -so real image- or negative -so fake image-.

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/figures/discriminator_architecture.png)

The adversarial on its turn is merely the discriminator built on top of the generator, but with the weights of the former frozen.

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/figures/adversarial_architecture.png)
<br/>
<br/>

# Training

A training step involves two phases, training the generator and training the discriminator, in the discriminator phase we train the discriminator for `n_critic` times, each time we feed the discriminator real images labelled $+1$, and fake images -synthesized by the generator- labelled $-1$, the role of the discriminator is to discriminate the images and determine the realness of the image, so it trains its weights to excel at this job.

On the contrary, the adverarial which takes noise 1D-vectors and transform them to images thanks to the generator and further feed them to the frozen discriminator along with the labels $+1$, is trying to take advantage of the discriminator yielding large loss values (because of the obviously fake images labelled as real), and by using the gradients of this loss the adversarial will try to change the direction of the generator's weights in such a way that it does generate more authentic images in the next step, and like that the generator is going to learn how to generate more plausible images, in summary the discriminator will be more and more capable of determining the  realness, and the generator is getting more and more dexter at synthesizing images that look almost real.

the loss is of course the Wasserstein loss and the optimizer is RMSprop.
this is the pseudo-code, taken from the original paper [1].

![alt text](https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation/blob/main/PseudoCode.PNG)
<br/>
<br/>

# How to use

To use train this GAN on your own dataset:

1 - clone the repository:
```
!git clone https://github.com/TheMagicShop/Wasserstein-GAN-for-Anime-Face-Generation.git

%cd Wasserstein-GAN-for-Anime-Face-Generation
```

2 - use this script to get help about parameters to use:
`!python train_generator.py -h`

3 - after deciding which parameters to change:
`!python train_generator.py --your_parameters your_values`

Note:

- if your images share the same size, and you know that size and want to keep it, use `--image_shape your_height your_width`, and ignore the `--resize` parameter.

- if you don't know the shape of your images, or you know but you want to resize them any why, please use `--resize  desired_height desired_width`, and ignore the `--image_shape` parameter.

after the training has finished, the generator of the format (.h5) will be saved in your directory for later use.

Alternatively, you can import the libraries:
```
from GANs import WassersteinGAN
from utils import DataPreparator
```
reading the code will give you an idea on how to initialize these classes, this gives an access to more functionalities like the methods, `save_weights()` to save the weights for later training, and `load_weights(path)` to load pretrained weights,
<br/>
<br/>

# References

[1] Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein GAN. https://arxiv.org/pdf/1701.07875v3.pdf

[2] Lilian Weng. From GAN to WGAN. https://lilianweng.github.io/posts/2017-08-20-gan/

[3] Rowel Atienza. Advanced Deep Learning with TensorFlow 2 and Keras, Second Edition, Chapter 5: Improved GANs.
