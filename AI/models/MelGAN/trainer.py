import time

import tensorflow as tf

from MelGAN import Discriminator, Generator


class MelGANTrain():
    def __init__(self, epochs=None, steps=None, checkpoint_dir='./Trained_Model'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps = steps
        self.checkpoint_dir = checkpoint_dir

    def compile(self, discriminator_opt, generator_opt):

        self.discriminator = Discriminator()
        self.generator = Generator()

        self.d_opt = discriminator_opt
        self.g_opt = generator_opt

        self.mse = tf.keras.losses.mse()
        self.mae = tf.keras.losses.mae()

        self.checkpoint = tf.train.Checkpoint(discriminator=self.discriminator,
                                              discriminator_opt=self.d_opt,
                                              generator = self.generator,
                                              generator_opt = self.g_opt)

    def train(self, datasets):
        for epoch in range(self.epochs):
            start = time.time()

            for batch in datasets:
                _train_step(batch)

            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(self.checkpoint_dir)

            print(f'Time for {epoch} is {time.time() - start} sec')



    def _train_step(self, batch):
        y, mels = batch
        y_hat = self._generator_step(y, mels)
        self._discriminator_step(y, y_hat)

    @tf.function()
    def _generator_step(self, y, mels):
        with tf.GradientTape as g_tape:
            y_hat = self.generator(mels)
            pred_hat = self.discriminator(y_hat)

            loss = 0

            for i in range(len(pred_hat)):
                loss += self.mse(pred_hat[i][-1], tf.ones_like(pred_hat[i][-1]))
            loss /= i + 1

            feature_matching_loss = 0

            _y = discriminator(tf.expand_dims(y, 2))
            for i in range(len(pred_hat)):
                for j in range(len(pred_hat[i]) - 1):
                    feature_loss += self.mae(pred_hat[i][j], _y[i][j])
            feature_matching_loss /= (i + 1) * (j + 1)

            loss += 10 * feature_matching_loss

            gradient = g_tape.gradient(loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(gradient, self.generator.trainable_variables))

    @tf.function()
    def _discriminator_step(self, y, y_hat):
        with tf.GradientTape as d_tape:
            y = tf.expand_dims(y, 2)
            pred = self.discriminator(y)
            pred_hat = self.discriminator(y_hat)

            real_loss = 0
            fake_loss = 0

            for i in range(len(pred)):
                real_loss += self.mse(pred[i][-1], tf.ones_like(pred[i][-1]))
                fake_loss += self.mse(pred_hat[i][-1], tf.ones_like(pred_hat[i][-1]))

            real_loss /= i + 1
            fake_loss /= i + 1

            loss = real_loss + fake_loss

            gradient = d_tape.gradient(loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(gradient, self.discriminator.trainable_variables))
