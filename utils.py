import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



class DataPreparator():
  def __init__(self, 
               real_source, 
               noise_dim,
               batch_size=64, 
               resize=None,
               to_grayscale=False,
               ):
    '''
    DataPreparator class object initialization, this utility will help us flow real 
    data directly from its directory, and generate fake images from a uniform distribution,
    in order to train the GAN.
    args:
      real_source: path of the real data folder.
      noise_dim: size of the sampled noise (uniform distribution).
      batch_size: batch_size.
      resize: whether to resize the original images, ex: (32, 32), default: None.
      to_grayscale: whether to transform images to grayscale, default: False.    
    '''
    self.real_source = real_source
    self.files = os.listdir(real_source)
    self.noise_dim = noise_dim
    self.batch_size = batch_size
    self.resize = resize
    self.to_grayscale = to_grayscale
    self.reset_index()
    self.generator = None

  def set_generator(self, generator):
    '''
    set an exterior generator, crucial to synthesize fake images.
    arg: generator (keras model)
    '''
    self.generator = generator
  
  def reset_index(self):
    '''to reset the indexes, starting from zero again'''
    self.start_index = 0
    self.end_index = self.batch_size
    random.shuffle(self.files) # shuffle to assure randomness
  
  def _update_index(self):
    '''update indexes, slide the window of the indexes by batch_size''''
    self.start_index += self.batch_size
    self.end_index += self.batch_size
    if self.end_index > len(self.files):
      # if the source is exhausted (all images are used), reset indexes and start again.
      self.reset_index()        

  @property
  def steps_per_epoch(self):
    '''how many steps in one cycle are there, before all images are used'''
    return len(self.files) // self.batch_size
  
  def prepare_real_batch(self):
    '''provide us with a batch of real images of size batch_size'''
    batch_files = self.files[self.start_index: self.end_index] # current window
    self._update_index() # update indexes for next use
    imgs = [] # store batch images here

    for f in batch_files:
      # path to the single image
      f_path = os.path.join(self.real_source, f)
      img = Image.open(f_path)
      if self.resize is not None:
        # resize if any
        img = img.resize(self.resize)
      if self.to_grayscale:
        # transform to grayscale if any
        img = img.convert('L')
      imgs.append(img)
    
    real_batch = np.stack(imgs) # size batch_size
    real_batch = real_batch.astype('float32') / 255. # normalize to [0, 1]
    return real_batch
  
  def prepare_fake_batch(self):
    '''prepare a batch of fake images, of size batch_size'''
    # inputs to generator should be sampled from a uniform distribution
    # and of size (batch_size, noise_dim)
    noise_batch = np.random.uniform(low=-1,
                                    high=1,
                                    size=(self.batch_size, self.noise_dim))
    fake_batch = self.generator.predict(noise_batch) # generate fake images
    return fake_batch

  def get_batch(self, phase):
    '''get a batch of data given the phase of training (discriminator, or adversarial)'''
    if self.generator is None:
      # generator should already be set in advance, raise error if it wasn't
      raise Exception('please set the generator using '\
                      'self.set_generator(generator)')
    
    # in WassersteinGAN labels are either 1 or -1
    ones = np.ones(shape=(self.batch_size, 1))
    if phase == 'discriminator':
      # if the discriminator is being trained, it takes image tensors as input
      # we need a real batch as well as a fake batch
      # real batch will be labeled as 1 (real), while the fake one will be labeled as -1 (fake)
      fake_batch = self.prepare_fake_batch()
      real_batch = self.prepare_real_batch()
      return {'real': (real_batch, ones),
              'fake': (fake_batch, -ones)}
    
    elif phase == 'adversarial':
      # if the adversarial is being trained, it only takes noise as input
      # we'll label them as 1 (real), to fool the discriminator which is frozen in this case
      noise = np.random.uniform(low=-1,
                                high=1,
                                size=[self.batch_size, self.noise_dim])
      return (noise, ones)
       
    else:
      raise Exception('please choose [phase] to bo either '\
                      'adversarial or discriminator')
    


def save_figure(name):
  '''
    utility to save a pyplot figure.
    arg: name of the figure
  '''
  os.makedirs('figures', exist_ok=True)
  name = name + '.png'
  dst = os.path.join('figures', name)
  plt.savefig(dst, bbox_inches='tight')
  print(f'saving {name} in ./figures ...')
