from __future__ import division

import tensorflow as tf
import numpy as np
import os
import time
from ops import *
from util import *
from progressbar import ETA, Bar, Percentage, ProgressBar


class CoopNets_video(object):

    def __init__(self, sess, config):

        self.sess = sess

        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.num_frames = config.num_frames
        self.num_chains = config.num_chains

        self.num_epochs = config.num_epochs

        # parameters for optimizer
        self.lr_gen = config.lr_gen
        self.beta1_gen = config.beta1_gen

        self.lr_des = config.lr_des
        self.beta1_des = config.beta1_des

        # parameters for descriptor type
        self.des_type = config.des_type
        self.gen_type = config.gen_type

        # parameters for inference
        self.refsig_gen = config.refsig_gen
        self.step_size_gen = config.step_size_gen
        self.sample_steps_gen = config.sample_steps_gen

        # parameters for data synthesis
        self.refsig_des = config.refsig_des
        self.step_size_des = config.step_size_des
        self.sample_steps_des = config.sample_steps_des

        # parameters for generator
        self.z_size_l = config.z_size_l
        self.z_size_h = config.z_size_h
        self.z_size_w = config.z_size_w
        self.z_size_dim = config.z_size_dim

        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.train_dir = os.path.join(self.output_dir, 'observed_sequence')
        self.sample_gen_dir = os.path.join(self.output_dir, 'synthesis_gen_sequence')
        self.sample_des_dir = os.path.join(self.output_dir, 'synthesis_des_sequence')
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.result_dir = os.path.join(self.output_dir, 'final_result')

        # testing
        # self.test_dir = os.path.join(self.output_dir, 'testing_GT_sequence')
        # self.testing_data_path = os.path.join(config.testing_data_path, config.category)

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

    def build_model(self):

        # declare placeholder
        self.z = tf.placeholder(shape=[None, self.z_size_l, self.z_size_h, self.z_size_w, self.z_size_dim], dtype=tf.float32, name='z')
        self.obs = tf.placeholder(shape=[None, self.num_frames, self.image_size, self.image_size, 3], dtype=tf.float32, name='obs')
        self.syn = tf.placeholder(shape=[None, self.num_frames, self.image_size, self.image_size, 3], dtype=tf.float32, name='syn')

        # build generator model
        self.gen_res = self.ST_generator(self.z, reuse=False)

        self.gen_loss = tf.reduce_mean(1.0 / (2 * self.refsig_gen * self.refsig_gen) * tf.square(self.obs - self.gen_res))
        self.gen_loss_mean, self.gen_loss_update = tf.contrib.metrics.streaming_mean(self.gen_loss)

        # optimizing generator
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen')]
        gen_optim = tf.train.AdamOptimizer(self.lr_gen, beta1=self.beta1_gen)
        gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_vars)
        self.apply_gen_grads = gen_optim.apply_gradients(gen_grads_vars)

        # symbolic langevin for generator
        self.langevin_ST_generator = self.langevin_dynamics_ST_generator(self.z)

        # ST descriptor
        obs_res = self.ST_descriptor(self.obs, reuse=False)
        syn_res = self.ST_descriptor(self.syn, reuse=True)

        self.des_loss = tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0))
        self.des_loss_mean, self.des_loss_update = tf.contrib.metrics.streaming_mean(self.des_loss)

        # optimizing descriptor
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('st_des')]
        des_optimizer = tf.train.AdamOptimizer(self.lr_des, beta1=self.beta1_des)
        des_grads_vars = des_optimizer.compute_gradients(self.des_loss, var_list=des_vars)
        self.apply_des_grads = des_optimizer.apply_gradients(des_grads_vars)

        # symbolic langevin for descriptor
        self.langevin_ST_descriptor = self.langevin_dynamics_ST_descriptor(self.syn)

        self.recon_err_mean, self.recon_err_update = tf.contrib.metrics.streaming_mean_squared_error(
            tf.reduce_mean(self.syn, axis=0),
            tf.reduce_mean(self.obs, axis=0))

    def ST_generator(self, inputs, reuse=False, is_training=True):

        with tf.variable_scope('gen', reuse=reuse):

            if self.gen_type == 'FC_S_64':

                convt1 = convt3d(inputs, (None, 14, 1, 1, 256), kernal=(5, 1, 1), strides=(1, 1, 1), padding="VALID",
                                 name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = tf.nn.relu(convt1)
                print convt1
                #40
                convt2 = convt3d(convt1, (None, 32, 32, 32, 128), kernal=(5, 32, 32), strides=(2, 1, 1), padding="VALID",
                                name="convt2")

                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = tf.nn.relu(convt2)
                print convt2

                convt3 = convt3d(convt2, (None, 64, 64, 64, 3), kernal=(5, 5, 5), strides=(2, 2, 2), padding="SAME",
                                 name="convt3")

                convt = tf.tanh(convt3)
                return convt

            elif self.gen_type == 'FC_S_128':

                convt1 = convt3d(inputs, (None, 14, 1, 1, 256), kernal=(5, 1, 1), strides=(1, 1, 1), padding="VALID",
                                 name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = tf.nn.relu(convt1)
                print convt1
                #40
                convt2 = convt3d(convt1, (None, 32, 32, 32, 128), kernal=(5, 32, 32), strides=(2, 1, 1), padding="VALID",
                                name="convt2")

                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = tf.nn.relu(convt2)
                print convt2

                convt3 = convt3d(convt2, (None, 64, 64, 64, 64), kernal=(5, 5, 5), strides=(2, 2, 2), padding="SAME",
                                 name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = tf.nn.relu(convt3)
                print convt3

                convt4 = convt3d(convt3, (None, 64, 128, 128, 3), kernal=(2, 5, 5), strides=(1, 2, 2), padding="SAME",
                                 name="convt4")

                convt = tf.tanh(convt4)
                return convt

            elif self.gen_type == 'ST':

                convt1 = convt3d(inputs, (None, 4, 4, 4, 512), kernal=(4, 4, 4), strides=(1, 1, 1), padding="SAME",
                                  name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = tf.nn.relu(convt1)
                print convt1

                convt2 = convt3d(convt1, (None, 8, 8, 8, 256), kernal=(4, 4, 4), strides=(2, 2, 2), padding="SAME",
                                 name="convt2")
                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = tf.nn.relu(convt2)
                print convt2

                convt3 = convt3d(convt2, (None, 16, 16, 16, 128), kernal=(4, 4, 4), strides=(2, 2, 2), padding="SAME",
                                 name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = tf.nn.relu(convt3)
                print convt3

                convt4 = convt3d(convt3, (None, 32, 32, 32, 64), kernal=(4, 4, 4), strides=(2, 2, 2), padding="SAME",
                                 name="convt4")
                convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
                convt4 = tf.nn.relu(convt4)

                convt5 = convt3d(convt4, (None, 64, 64, 64, 3), kernal=(4, 4, 4), strides=(2, 2, 2), padding="SAME",
                                 name="convt5")
                convt5 = tf.tanh(convt5)
                return convt5
            else:
                return NotImplementedError


    def langevin_dynamics_ST_generator(self, z_arg):
        def cond(i, z):
            return tf.less(i, self.sample_steps_gen)

        def body(i, z):
            noise = tf.random_normal(shape=tf.shape(z), name='noise')
            gen_res = self.ST_generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.refsig_gen * self.refsig_gen) * tf.square(self.obs - gen_res), axis=0)

            grad = tf.gradients(gen_loss, z, name='grad_gen')[0]

            z = z - 0.5 * self.step_size_gen * self.step_size_gen * (z + grad) + self.step_size_gen * noise
            return tf.add(i, 1), z

        with tf.name_scope("langevin_dynamics_generator"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z_arg])
            return z


    def ST_descriptor(self, inputs, reuse=False):

        with tf.variable_scope('st_des', reuse=reuse):

            """
            This is the spatial fully connected model used for synthesizing dynamic textures with only temporal
            stationarity
            """

            if self.des_type == "FRAME_3":

                conv1 = conv3d(inputs, 120, (1, 3, 3), strides=(1, 2, 2), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 200, (3, 3, 3), strides=(2, 2, 2), padding="SAME", name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 1, (conv2.shape[1], conv2.shape[2], conv2.shape[3]), strides=(2, 2, 2),
                                 padding=(0, 0, 0), name="conv3")

                return conv3

            elif self.des_type == "FRAME_2":

                conv1 = conv3d(inputs, 200, (3, 7, 7), strides=(1, 2, 2), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 10, (conv1.shape[1], conv1.shape[2], conv1.shape[3]), strides=(2, 2, 2),
                               padding=(0, 0, 0), name="conv2")

                return conv2

            elif self.des_type == "FC_S":

                conv1 = conv3d(inputs, 120, (5, 5, 5), strides=(2, 2, 2), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 30, (5, conv1.shape[2], conv1.shape[3]), strides=(2, 1, 1), padding=(2, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 10, (5, 1, 1), strides=(2, 1, 1), padding=(2, 0, 0), name="conv3")

                return conv3

            else:

                return NotImplementedError

    def langevin_dynamics_ST_descriptor(self, syn_arg):
        def cond(i, syn):
            return tf.less(i, self.sample_steps_des)

        def body(i, syn):
            noise = tf.random_normal(shape=tf.shape(syn), name='noise')
            syn_res = self.ST_descriptor(syn, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.step_size_des * self.step_size_des * (syn / self.refsig_des / self.refsig_des - grad)
            syn = syn + self.step_size_des * noise
            return tf.add(i, 1), syn

        with tf.name_scope("langevin_dynamics"):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg])

            return syn

    def train(self):

        # build dynamic generator model
        self.build_model()

        # Prepare training data
        loadVideoToFrames(self.data_path, self.train_dir)
        train_data = getTrainingData(self.train_dir, num_frames=self.num_frames, image_size=self.image_size, scale_method='tanh')
        num_batches = int(math.ceil(train_data.shape[0] / self.batch_size))

        print(train_data.shape)

        # initialize training
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)

        gen_syn = np.zeros((self.num_chains * num_batches, train_data.shape[1], train_data.shape[2], train_data.shape[3], train_data.shape[4]))
        des_syn = np.zeros((self.num_chains * num_batches, train_data.shape[1], train_data.shape[2], train_data.shape[3], train_data.shape[4]))

        for epoch in xrange(self.num_epochs):

            for iBatch in xrange(num_batches):

                indices_obs_batch = slice(iBatch * self.batch_size,
                                      min(train_data.shape[0], (iBatch + 1) * self.batch_size))

                current_batch_size = min(train_data.shape[0], (iBatch + 1) * self.batch_size) - iBatch * self.batch_size


                start_time = time.time()

                batch_obs = train_data[indices_obs_batch, :, :, :, :]

                # Step G0: generate hidden variables
                batch_z = np.random.normal(0, 1, size=(self.num_chains, self.z_size_l, self.z_size_h, self.z_size_w, self.z_size_dim)) #* self.refsig_gen

                # generate data
                recon = self.sess.run(self.gen_res, feed_dict={self.z: batch_z})

                # Step D1: obtain synthesized data Y
                syn = self.sess.run(self.langevin_ST_descriptor, feed_dict={self.syn: recon})

                # Step G1: Update hidden variables using synthesized data as training data (inference by Langevin)

                # batch_z, batch_content_vectors, batch_motion_type_vectors = self.sess.run(self.langevin_conditional_dyn_generator,
                #                         feed_dict={self.truncated_batch_z_placeholder: batch_z,
                #                                    self.batch_first_frame_placeholder: batch_first_frame,
                #                                    self.truncated_batch_obs_placeholder: syn,
                #                                    self.batch_content_placeholder: batch_content_vectors,
                #                                    self.batch_motion_type_placeholder: batch_motion_type_vectors})

                # Step D2: update descriptor
                self.sess.run([self.des_loss, self.des_loss_update, self.apply_des_grads],
                              feed_dict={self.syn: syn, self.obs: batch_obs})

                # # store the last state of the batch as the first stage of next truncation
                # batch_last_state = self.sess.run(self.next_state,
                #                                  feed_dict={self.truncated_batch_z_placeholder: batch_z,
                #                                             self.batch_state_initial_placeholder: batch_state_initial})

                # Step G2: update dyn generator
                self.sess.run([self.gen_loss, self.gen_loss_update, self.apply_gen_grads],
                              feed_dict={self.z: batch_z, self.obs: syn})


                # z[indices_batch, indices_truncation, :] = batch_z

                indices_syn_batch = slice(iBatch * self.num_chains, (iBatch + 1) * self.num_chains)
                gen_syn[indices_syn_batch, :, :, :, :] = recon
                des_syn[indices_syn_batch, :, :, :, :] = syn

                [gen_loss_avg, des_loss_avg] = self.sess.run([self.gen_loss_mean, self.des_loss_mean])

                end_time = time.time()
                print(
                'Epoch #%d of #%d, batch #%d of #%d, generator loss: %4.4f, descriptor loss: %4.4f, time: %.2fs' % (
                    epoch + 1, self.num_epochs, iBatch + 1, num_batches, gen_loss_avg, des_loss_avg, end_time - start_time))


            if epoch % self.log_step == 0:
                if not os.path.exists(self.sample_gen_dir):
                    os.makedirs(self.sample_gen_dir)

                #saveSampleSequence(gen_syn + data_mean, self.sample_gen_dir, epoch, col_num=10,
                #                   scale_method='original')
                saveSampleSequence(gen_syn, self.sample_gen_dir, epoch, col_num=10,
                                   scale_method='tanh')

                if not os.path.exists(self.sample_des_dir):
                    os.makedirs(self.sample_des_dir)

                saveSampleSequence(des_syn, self.sample_des_dir, epoch, col_num=10,
                                   scale_method='tanh')
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(self.sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

            if epoch % 20 == 0:

                saveSampleVideo(des_syn, self.result_dir, original=(train_data),
                                global_step=epoch,
                                scale_method='tanh')

    def test(self, ckpt):

        assert ckpt is not None, 'no checkpoint provided.'

        sample_dir_testing = os.path.join(self.output_dir, 'synthesis_sequence_testing')
        result_dir_testing = os.path.join(self.output_dir, 'final_result_testing')

        # self.num_batches_generated = 1
        self.batch_size_generated = self.num_chains
        self.num_frames_generated = self.truncated_backprop_length

        self.truncated_batch_z_placeholder_testing = tf.placeholder(
            shape=[None, self.num_frames_generated, self.z_size],
            dtype=tf.float32, name='z_testing')

        self.batch_state_initial_placeholder = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32,
                                                              name='state_initial')
        self.batch_content_placeholder = tf.placeholder(shape=[None, self.content_size], dtype=tf.float32,
                                                        name='content')
        self.batch_motion_type_placeholder = tf.placeholder(shape=[None, self.motion_type_size], dtype=tf.float32,
                                                            name='motion_type')

        images_syn , next_state, all_states = self.dyn_generator(self.truncated_batch_z_placeholder_testing,
                                                    self.batch_state_initial_placeholder, self.batch_content_placeholder,
                                                    self.batch_motion_type_placeholder, reuse=False)

        # sample_videos = np.random.randn(self.num_batches_generated * self.batch_size_generated, self.num_frames_generated, self.image_size, self.image_size, 3)

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        # sample_z = np.random.normal(0, 1, size=(self.sample_size_test, self.z_size * self.z_size_d * self.z_size_h * self.z_size_w)) * self.refsig

        z = np.random.normal(0, 1,
                             size=(self.batch_size_generated, self.num_frames_generated, self.z_size)) * self.refsig_gen

        state_initial = np.random.normal(0, 1, size=(self.batch_size_generated, self.state_size))

        content = np.random.normal(0, 1, size=(self.batch_size_generated, self.content_size))
        motion_type = np.random.normal(0, 1, size=(self.batch_size_generated, self.motion_type_size))

        recon = self.sess.run(images_syn, feed_dict={self.truncated_batch_z_placeholder_testing: z,
                                                     self.batch_state_initial_placeholder: state_initial,
                                                     self.batch_content_placeholder: content,
                                                     self.batch_motion_type_placeholder: motion_type})


        # saveSampleSequence(recon.reshape((1, recon.shape[0], self.image_size, self.image_size, 3)), self.sample_dir,  epoch, col_num=10, scale_method='tanh')
        saveSampleSequence(recon, sample_dir_testing, col_num=10, scale_method='tanh')
        saveSampleVideo(recon, result_dir_testing, scale_method='tanh')
