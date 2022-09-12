from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Reshape, Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import plot_model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os
import pathlib
from utils import save_figure


'''
Wasserstein GAN is simply the Vanilla GAN except that the loss function is
based on Wasserstein 1 (or Earth Mover's Distance), more informations
about the theory behind WGANs can be found here, 
https://lilianweng.github.io/posts/2017-08-20-gan/
original paper: https://arxiv.org/pdf/1701.07875.pdf 
'''

class WassersteinGAN():
  def __init__(self, 
               data_preparator,
               image_shape=(64, 64, 3),
               noise_dim=128,
               learning_rate=5e-5,
               decay_rate=1,
               save_path='./'
               ):
    
    '''
    initialization of the WassersteinGAN class,
    args:
      data_preparator: DataPreparator object, pre-built class in utils.py from which
        we are going to flow data directly from the library.
      image_shape: the desired shape genearted images (basically the shape of the real images)
      noise_dim: the dimension of the space (uniform distribution) from which we 
        are going to sample points, feed it to generator and synthesize new fake images.
      learning_rate: the learning rate at which the discriminator trains, and half of which
        the adversarial trains.
      decay_rate: the rate by which the learning_rate will decline after each 1000 steps.
      save_path: path to save parameters and figures.
    '''
    self.learning_rate = learning_rate
    self.decay_rate = decay_rate
    self.noise_dim = noise_dim
    
    # generator uses Conv2DTranspose to upsample tensor's width and height
    # and reduce the feature maps, ex: 4x4x512 -> 8x8x256
    # strides will be 2, so we're going to take as much 2's as we can (division)
    # from the image width and height until we reach a very small feature map
    # of around 4x4
    self.n_upsamples = 0
    a, b = image_shape[:2]
    while a > 7 and b > 7 and a%2 == 0 and b%2 == 0:
      a, b = a // 2, b // 2
      self.n_upsamples += 1      # trying to downsample until around 4x4
    if a > 7 or b > 7: # if we got sth like (x*2^n, y*2^n) where x or y is greater than 7
      raise Exception("please try to find an image size of the form "\
                      "(x*2^n, y*2^n) where x and y are in (1, 3, 5, 7)")
    # we are going to use as much conv layers as n_upsamples, 
    # plus two others with strides=1 (no upsampling)
    n_conv_layers = self.n_upsamples + 2  # last two conv layers have strides=1
    self.layer_filters = [image_shape[-1]] + \
      [min(32*2**i, 512) for i in range(n_conv_layers - 1)] 
    self.layer_filters = self.layer_filters[::-1]   # [512, ..., 64, 32, 3]
    # of course the discriminator use the reverse (downsampling)
    self.kernel_size = 5 # kernel size of 5

    # build the models and compile 
    self.generator = self._build_generator(image_shape=image_shape,
                                           noise_dim=noise_dim) 
    self.discriminator = self._build_discriminator(image_shape=image_shape)
    self.adversarial = self._build_adversarial(noise_dim=noise_dim)
    
    # store the weights of all models here
    self._weights = None
    
    # the DataPreparator object, and set the generator to be able to provide
    # us with fake data along with the real one.
    self.data_preparator = data_preparator
    self.data_preparator.set_generator(self.generator)
    
    # use a specifique data points (noise) from which we'll generate fake images,
    # and keep an eye on the progress of the WGAN
    self.tracked_noise = np.random.uniform(low=-1,
                                           high=1,
                                           size=(64, self.noise_dim))
    
    # some paths to save parameters and figures
    save_path = pathlib.Path(save_path)
    self.save_path = save_path
    path_params = save_path / 'params'
    path_tracked = save_path / 'figures' / 'tracked_noise'
    os.makedirs(path_params, exist_ok=True)
    os.makedirs(path_tracked, exist_ok=True)

    # save losses
    self.losses = dict(discriminator=[], 
                       adversarial=[])

  @property
  def weights(self):
    '''weights getter, [dictionnary]'''
    self._weights = {'generator': self.generator.get_weights(),
                     'discriminator': self.discriminator.get_weights(),
                     'adversarial': self.adversarial.get_weights()}
    return self._weights
  
  @weights.setter
  def weights(self, kw):
    '''weights setter'''
    self._weights = kw
  
  def save_weights(self, path):
    '''
    save the pickled weights dictionnary to path.
    arg: path.
    '''
    with open(path, 'wb') as f:
      pickle.dump(self.weights, f)
  
  def load_weights(self, path):
    '''
    load the weights dictionnary from path.
    arg: path
    '''
    with open(path, 'rb') as f:
      self.weights = pickle.load(f)
    # apply the weights on the models (set_weights)
    self.apply_weights()
  
  def apply_weights(self,):
    '''set the current weights to the models'''
    for model in (self.generator,
                  self.discriminator,
                  self.adversarial):
        model.set_weights(self.weights[model.name])
  
  def save_generator(self, name=None):
    '''
    save the generator (.h5 file), for later usage.
    arg: name 
    '''
    if name is None:
      name = 'generator'
    path = self.save_path / 'params' / f'{name}.h5'
    # delete last version and save the new one. 
    self.generator.save(path, overwrite=True)

  def _build_generator(self,
                       image_shape,
                       noise_dim,
                       ):
    '''
    construct the generator, starting with a Dense layer to transform input tensors
    of dimension noise_dim into tensors of a sufficent dimensionality to reshape them 
    into small feature maps, then use a certain number of Conv2DTranspose
    (preceded by BatchNormalizaion and ReLU activation) to upsample
    the feature maps to attain the image_shape.
    args:
      image_shape: desired shape of generated images (also the shape of real images)
      noise_dim: dimension of input data points (uniform distribution)
    '''
    print('Building the generator...')
    # the shape of the first tensor, after Dense layer, and before upsampling.
    first_shape = (image_shape[0] // 2 ** self.n_upsamples, 
                   image_shape[1] // 2 ** self.n_upsamples,
                   self.layer_filters[0])
    
    inputs = Input(shape=(noise_dim, ), name='generator_input')
    x = Dense(first_shape[0] * first_shape[1] * first_shape[2])(inputs)
    x = Reshape(first_shape)(x)  # ~4x4x512
    
    # size of the filter decreases until 3, to achieve (?, ?, 3), -three channels-
    layer_filters = self.layer_filters
    strides = 2
    for filters in layer_filters:
      if filters <= layer_filters[-2]:
        # last two Conv layers use only strides=1 (no upsampling)
        strides = 1
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(filters=filters,
                          kernel_size=self.kernel_size,
                          strides=strides,
                          padding='same')(x)
    
    # apply a sigmoidal activation at the end, to squash the images pixel values
    # inside [0, 1]
    outputs = Activation('sigmoid', name='generator_ouput')(x)
    generator =  Model(inputs=inputs,
                       outputs=outputs,
                       name='generator')
    
    return generator

  def _build_discriminator(self,
                           image_shape):
    '''
    construct the discriminator, it basically does the reverse operations of what the 
    generator has done, namely Conv2D (preceded with LeakyReLU) with self.layer_filters
    flipped this time, the filter size increases while downsampling (strides=2),
    except last layer where strides=1, all this is followed by a Flatten then a Dense
    layer to classify image as real (+1) or fake (-1), with a linear activation to conform 
    with the wasserstein loss.
    arg: image_shape.
    '''
    print('Building the discriminator...')
    inputs = Input(shape=image_shape, name='discriminator_input')
    # input tensor of size (batch_size, height, width, channels)
    x = inputs
    strides = 2
    layer_filters = self.layer_filters[-2::-1] # [32, 64, ..., 512]
    for filters in layer_filters:
      if filters == layer_filters[-1]:
        # last conv layer uses strides=1
        strides = 1
      x = LeakyReLU(alpha=0.2)(x)
      x = Conv2D(filters=filters,
                 kernel_size=self.kernel_size,
                 strides=strides,
                 padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    outputs = Activation('linear', name='discriminator_output')(x)
    discriminator = Model(inputs=inputs, 
                          outputs=outputs,
                          name='discriminator')
    
    # learning_rate will decline exponentially after each 1000 steps,
    # with decay rate, default=1
    schedule = ExponentialDecay(initial_learning_rate=self.learning_rate,
                                decay_rate=self.decay_rate,
                                decay_steps=1000)
    # use RMSprop unlike the paper which uses Adam
    optimizer = RMSprop(learning_rate=schedule)
    # the loss is the wasserstein loss
    discriminator.compile(loss=self._wasserstein_loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    
    return discriminator
  
  def _build_adversarial(self,
                         noise_dim):
    '''
    build the adversarial which is simply the discriminator built on top of the generator,
    while the weights of the discriminator are forzen, the adversarial takes noise as input,
    flow it through the generator to generate fake images, along with the labels of +1,
    (labeled as real) the generator will get penalized beacuse the discriminator will
    classify them (images) as fake and give a high loss which is used to tune the generator parameters
    so it'll do a better job in the next step.
    arg: noise_dim.
    '''
    print('Building the adversarial...')
    # freeze the discriminator
    self.discriminator.trainable = False
    # input is a tensor of the form (batch_size, noise_dim)
    inputs = Input(shape=(noise_dim, ), name='adversarial_input')
    adversarial = Model(inputs=inputs,
                        outputs=self.discriminator(
                            self.generator(inputs)),
                        name='adversarial'
                        )
    
    # to assure that the generator doesn't prematurely overcome the discriminator
    # in terms of learning progress, we're going to use the half of the learning_rate
    # used in discriminator, note that this specific proportion is drawn solely
    # based on trial and error and is exclusive to the dataset of Anime Faces
    schedule = ExponentialDecay(initial_learning_rate=self.learning_rate / 2,
                                decay_rate=self.decay_rate,
                                decay_steps=1000)
    optimizer = RMSprop(learning_rate=schedule)
    adversarial.compile(loss=self._wasserstein_loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    
    return adversarial
  
  def _wasserstein_loss(self, y_true, y_pred):
    '''
    wasserstein loss is used to judge the true labels from the fake ones,
    this is the ultimate simplified version of it,
    refer to the papaer above for more information.
    args:
      y_true: true labels.
      y_pred: predicted labels
    '''
    return -tf.reduce_mean(y_true * y_pred)

  def _train_batch_discriminator(self, clip_value):
    '''
    train the discriminator for one batch, given real and fake images labeled +1 and -1
    respectively, the discriminator will train to classify the realness of the images.
    arg: clip value 
    '''
    # get one batch of data from the data_preparator object
    # then train individually on real images then on fake images
    data = self.data_preparator.get_batch(phase='discriminator')
    m_real = self.discriminator.train_on_batch(*data['real'])
    m_fake = self.discriminator.train_on_batch(*data['fake'])
    # average the metrics out of the real and fake
    metrics = tuple((m_real[i] + m_fake[i]) / 2 for i in [0, 1])
    
    # apply the lipschitz trick corresponding to the wasserstein loss 
    # clip weights into a very narrow compact space
    for layer in self.discriminator.layers:
      weights = layer.get_weights()
      weights = [np.clip(w, -clip_value, clip_value) \
                  for w in weights]
      layer.set_weights(weights)

    return metrics
  
  def _train_batch_adversarial(self):
    '''
    while the weights of the discriminator are forzen, the adversarial takes noise as input,
    flow it through the generator to generate fake images, along with the labels of +1,
    (labeled as real) the generator will get penalazied beacuse the discriminator will
    classify them (images) as fake and give a high loss which is used to tune the generator parameters
    so it'll do a better job in the next step.
    '''
    # get a noise batch along with the labels +1
    data = self.data_preparator.get_batch(phase='adversarial')
    metrics = self.adversarial.train_on_batch(*data)
    return metrics
  
  def train(self, 
            n_critic=5,
            clip_value=0.01,
            train_steps=20000,
            step_0=0,
            save_interval=500,
            save_best_only=True
            ):
    '''
    the essential function, to train the GAN, we train the discriminator for n_critic
    times, each for one batch, then train the adversarial one time for one batch and repeat.
    args:
      n_critic: number of time the discriminator will be trained in one step.
      clip_value: clip weights inside [-value, +value] to apply the lipschitz condition.
      step_0: (optional) step at which training will begin (in case of pretrained weights).
      save_interval: the frequency at which we are going to save parameters and figures.
      save_best_inly: save solely best weights after each save_interval.
    '''
    self.step_0 = step_0
    for step in range(step_0, step_0 + train_steps):
      loss, acc = 0, 0
      # train the discriminator for n_critic times and save logs
      for _ in range(n_critic):
        # train the discriminator on one batch
        metrics = self._train_batch_discriminator(clip_value)
        # sum up with a fraction of metrics 
        loss += metrics[0]/n_critic
        acc += metrics[1]/n_critic
      log = f"step: {step}, disc_loss: {loss:.3f}"
      self.losses['discriminator'].append(loss)
      
      # train the adversarial for one batch and save logs
      loss, acc = self._train_batch_adversarial()
      log = f"{log}, adv_loss: {loss:.3f}"
      self.losses['adversarial'].append(loss)
      
      # display the logs
      print(log)
      
      # if step is a multiplier of save_interval
      if (step + 1) % save_interval == 0:
        #save weights
        if save_best_only:
          path = self.save_path / 'params' / f'weights.pkl'
        else:
          path = self.save_path / 'params' / f'weights_step_{step+1}.pkl'
        self.save_weights(path)
        # save tracked noise generated images
        to_path = self.save_path / 'figures' / 'tracked_noise' \
                                / f'tracked_noise_step_{step+1}.png' 
        self.track_noise(to_path=to_path, show=True)
        # save the generator
        self.save_generator()
  
  def track_noise(self, 
                  noise=None, 
                  n_rows=8,
                  n_cols=8,
                  show=False,
                  to_path=None):
    '''
    this function is to be used at any point of training (or after)
    to synthesize fake images from the earlier created noise.
    args:
      noise: of size (n_points, noise_dim), the input of the generator.
      n_rows: number of rows in the subplots.
      n_cols: number of columns in the subplots.
      show: to show the subplots in the output or not.
      to_path: path to save the figure to.
    '''
    if noise is None:
      noise = self.tracked_noise
    if len(noise) < n_rows*n_cols:
      raise Exception('noise data size is less than n_rows*n_cols')
    
    # make the images out of the noise
    imgs = self.generator.predict(noise)
    # initialize the figure with the subplots
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2*n_cols, 2*n_rows))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, ax in enumerate(axes.ravel()):
      ax.imshow(imgs[i])
      ax.set_axis_off()
    if to_path is not None:
      plt.savefig(to_path, bbox_inches='tight')
    # show or don't
    if show:
      plt.show()
    else:
      plt.close('all')

  def plot_models(self):
    ''' utility to save the model architectures '''
    # save to this path 
    to_file = self.save_path / 'figures'
    # crate the folder if doesn't exist
    os.makedirs(to_file, exist_ok=True)

    print(f'plotting model architectures on {to_file}/ ...')
    plot_model(self.generator,
               show_shapes=True,
               to_file=to_file / 'generator_architecture.png')
    plot_model(self.discriminator, 
               show_shapes=True,
               to_file=to_file / 'discriminator_architecture.png')
    plot_model(self.adversarial, 
               show_shapes=True,
               to_file=to_file / 'adversarial_architecture.png')

  def plot_loss(self, show=False):
    '''
    plot loss curves after the training has been ended.
    arg: show [bool] (show or just save)
    '''
    # if step_0 is not 0, then tho bounds are different than [0, train_steps]
    xmin, xmax = self.step_0, self.step_0 + len(self.losses['discriminator'])
    # capture the corresponding x's (xmin-xmax is negative)
    disc_loss = self.losses['discriminator'][xmin-xmax:] 
    adv_loss = self.losses['adversarial'][xmin-xmax:]
    # initialize the figure
    fig = plt.figure(figsize=(16, 4))
    plt.plot(range(xmin, xmax), disc_loss, 'r-', label='discriminator loss')
    plt.plot(range(xmin, xmax), adv_loss, 'b-', label='adversarial loss')
    plt.xlabel('step')
    plt.ylabel('wasserstein loss')
    plt.title('discriminator and adversarial loss curves')
    plt.legend(loc='best')
    save_figure(f'losses_{xmin}_{xmax}')
    # show or don't
    if show:
      plt.show()
    else:
      plt.close('all')
