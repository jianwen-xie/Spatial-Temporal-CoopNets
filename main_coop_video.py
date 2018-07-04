import argparse
import tensorflow as tf
from src.CoopNets_video import CoopNets_video

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CONFIG = tf.app.flags.FLAGS

# model hyper-parameters
tf.flags.DEFINE_integer('image_size', 128, 'Image size to rescale images')

# training hyper-parameters
tf.flags.DEFINE_integer('num_epochs', 1500, 'Number of epochs') # 1500
tf.flags.DEFINE_integer('num_frames', 64, 'number of frames used in training data')
tf.flags.DEFINE_integer('batch_size', 23, 'number of training examples (videos) in each batch')

tf.flags.DEFINE_integer('num_chains', 2, 'number of synthesized results for each batch of training')


# parameters for generator (latent variables)
tf.flags.DEFINE_string('gen_type', 'FC_S_128', 'generator type')

tf.flags.DEFINE_integer('z_size_dim', 5, 'channel of latent variables')  #5
tf.flags.DEFINE_integer('z_size_l', 10, 'length of latent variables')     #19
tf.flags.DEFINE_integer('z_size_h', 1, 'height of latent variables')     #19
tf.flags.DEFINE_integer('z_size_w', 1, 'width of latent variables')     #19

tf.flags.DEFINE_float('lr_gen', 0.0002, 'learning rate for generator')  # 0.003
tf.flags.DEFINE_float('beta1_gen', 0.5, 'momentum term in Adam for generator')  # 0.5

tf.flags.DEFINE_float('refsig_gen', 1, 'sigma')  # 0.003
tf.flags.DEFINE_float('step_size_gen', 0.003, 'delta/ step size for langevin for generator') # 0.003
tf.flags.DEFINE_integer('sample_steps_gen', 5, 'number of steps of Langevin sampling in generator')

# parameters for descriptor
tf.flags.DEFINE_string('des_type', 'FC_S', 'descriptor type')

tf.flags.DEFINE_float('lr_des', 0.01, 'learning rate for descriptor')
tf.flags.DEFINE_float('beta1_des', 0.5, 'momentum term in Adam for descriptor')

tf.flags.DEFINE_float('refsig_des', 0.016, 'standard deviation of the reference distribution ')  # 0.003
tf.flags.DEFINE_float('step_size_des', 0.002, 'delta/ step size for langevin for descriptor') # 0.05
tf.flags.DEFINE_integer('sample_steps_des', 10, 'number of steps of Langevin sampling in descriptor')

# misc
tf.flags.DEFINE_string('output_dir', './output_coop_video', 'output directory')
tf.flags.DEFINE_string('category', 'fire_pot', 'name of category')
tf.flags.DEFINE_string('data_path', '../trainingVideo/data_synthesis', 'path of the training data')
tf.flags.DEFINE_integer('log_step', 10, 'number of steps to output synthesized image')

# testing
tf.flags.DEFINE_string('sample_size_test', 10, 'total number of samples generated in testing')
tf.flags.DEFINE_string('batch_size_test', 10, 'number of samples generated in each batch in testing')

#tf.flags.DEFINE_string('testing_data_path', '../trainingVideo/data_synthesis', 'path of the training data')

def main():
    with tf.Session() as sess:
        model = CoopNets_video(sess, CONFIG)
        model.train()

        #self.log_dir = os.path.join(self.output_dir, 'log')
        #model.test('./output_coop_dyn_generator/cows/model/model.ckpt-1990')


if __name__ == '__main__':
    main()
