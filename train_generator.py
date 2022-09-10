from utils import DataPreparator
from GANs import WassersteinGAN
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    
    my_help = 'data path (source), should only contain images'
    parser.add_argument('--source', type=str, help=my_help)
    my_help = 'the shape of your images (height, width), ex: 64 64,'\
              ' if you dont know the size of your images or whether'\
              ' they have a common shape or not, please use'\
              ' --resize= x y, where x and y are the desired height and width'
    parser.add_argument('--shape', nargs='+', type=int, help=my_help)
    my_help = 'to resize images, useful if they dont have'\
              ' a common size, ex: 64 64, if --shape is not provided'\
              ' and --resize is not provided, then a shape of (64, 64)'\
              ' will automatically be used'
    parser.add_argument('--resize', nargs='+', type=int, help=my_help)
    my_help = 'to convert images to grayscale, or if you have grayscale images, default: False'
    parser.add_argument('--to_grayscale', type=bool, default=False, help=my_help)
    my_help = 'batch_size, default: 64'
    parser.add_argument('--batch_size', type=int, default=64, help=my_help)
    my_help = 'the dimension of a uniform distribution from' \
              ' which the images will be generated, default: 128'
    parser.add_argument('--noise_dim', type=int, default=128, help=my_help)
    my_help = 'the learning rate at which learning will be performed, default: 1e-5'
    parser.add_argument('--learning_rate', default=1e-5, type=float, help=my_help)
    my_help ='the decay rate, at which learning rate will decline after'\
             ' each 1000 steps, ex: 0.99, default: 1'
    parser.add_argument('--decay_rate', type=float, default=1., help=my_help)
    my_help = 'n_critic, training the discriminator n_critic times,'\
              ' before training the adversarial, for every step, default: 5'
    parser.add_argument('--n_critic', type=int, default=5, help=my_help)
    my_help = 'clipping value, weights will be clipped after each discriminator'\
              ' training step, default: 0.01'
    parser.add_argument('--clip_value', type=float, default=0.01, help=my_help)
    my_help = 'taining steps, default: 20000'
    parser.add_argument('--train_steps', type=int, default=20000, help=my_help)
    my_help = 'step_0 -optional-, useful if you provide an already trained weights'
    parser.add_argument('--step_0', type=int, default=0, help=my_help)
    my_help = 'the saving path to which we are going to save plots and parameters'\
              ' if None, current directory will be used'
    parser.add_argument('--save_path', type=str, default='./', help=my_help)
    my_help = 'saving parameters and plots after each save_interval, default: 500'
    parser.add_argument('--save_interval', type=int, default=500, help=my_help)
    my_help = 'overwriting the last saved weights, defualt: True'
    parser.add_argument('--save_best_only', type=bool, default=True, help=my_help)
    my_help = 'optional, to set an already pretrained weights, should be a path to an pkl file'
    parser.add_argument('--load_weights', type=str, default=None, help=my_help)
    
    args = parser.parse_args()
    
    if args.shape is None and args.resize is None:
        image_shape = (64, 64, 3)
        resize = (64, 64)
    elif args.resize is not None:
        resize = tuple(args.resize)
        image_shape = resize + (3, )
    else:
        image_shape = tuple(args.shape) + (3, )
        resize = None
        
    real_source = args.source
    to_grayscale = args.to_grayscale
    batch_size = args.batch_size
    noise_dim = args.noise_dim
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    n_critic = args.n_critic
    clip_value = args.clip_value
    train_steps = args.train_steps
    step_0 = args.step_0
    save_path = args.save_path
    save_interval = args.save_interval
    save_best_only = args.save_best_only
    weights_path = args.load_weights
    
    data_preparator = DataPreparator(real_source=real_source,
                                     noise_dim=noise_dim,
                                     batch_size=batch_size,
                                     resize=resize,
                                     to_grayscale=to_grayscale)
    
    wgan = WassersteinGAN(data_preparator=data_preparator,
                          image_shape=image_shape,
                          noise_dim=noise_dim,
                          learning_rate=learning_rate,
                          decay_rate=decay_rate,
                          save_path=save_path)
    if weights_path is not None:
        wgan.load_weights(weights_path)
    
    wgan.plot_models()
    wgan.train(n_critic=n_critic,
               clip_value=clip_value,
               train_steps=train_steps,
               step_0=step_0,
               save_interval=save_interval,
               save_best_only=save_best_only)
    wgan.plot_loss()
    
