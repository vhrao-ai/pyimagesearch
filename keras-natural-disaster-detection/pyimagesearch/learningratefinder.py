# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile


# ------------------------
#   LearningRateFinder
# ------------------------
class LearningRateFinder:
    def __init__(self, model, stop_factor=4, beta=0.98):
        # store the model, stop factor, and beta value (for computing a smoothed, average loss)
        self.model = model
        self.stop_factor = stop_factor
        self.beta = beta
        # initialize the list of learning rates and losses respectively
        self.lrs = []
        self.losses = []
        # initialize the learning rate multiplier, average loss, best loss found thus far, current batch number
        # and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # re-initialize all variables from the constructor
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        # define the set of class types that we will be checking
        iter_classes = ["NumpyArrayIterator", "DirectoryIterator", "DataFrameIterator", "Iterator", "Sequence"]
        # return whether our data is an iterator
        return data.__class__.__name__ in iter_classes

    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        # grab the loss at the end of this batch, increment the total number of batches processed, compute the average
        # loss, smooth it and update the losses list with the smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * 1)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)
        # compute the maximum loss stopping factor value
        stop_loss = self.stop_factor * self.bestLoss
        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth < self.bestLoss:
            self.bestLoss = smooth
        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, train_data, start_lr, end_lr, epochs=None, steps_per_epoch=None, batch_size=32,
             sample_size=2048, verbose=1):
        # reset the class-specific variables
        self.reset()
        # determine if we are using a data generator or not
        use_gen = self.is_data_iter(train_data)
        # if we're using a generator and the steps per epoch is not supplied, raise an error
        if use_gen and steps_per_epoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)
        # if we're not using a generator then our entire dataset must already be in memory
        elif not use_gen:
            # grab the number of samples in the training data and then derive the number of steps per epoch
            num_samples = len(train_data[0])
            steps_per_epoch = np.ceil(num_samples / float(batch_size))
        # if no number of training epochs are supplied, compute the training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sample_size / float(steps_per_epoch)))
        # compute the total number of batch updates that will take place while we are attempting to find a good starting
        # learning rate
        num_batch_updates = epochs * steps_per_epoch
        # derive the learning rate multiplier based on the ending learning rate, starting learning rate, and
        # total number of batch updates
        self.lrMult = (end_lr / start_lr) ** (1.0 / num_batch_updates)
        # create a temporary file path for the model weights and then save the weights (so we can reset the weights
        # when we are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)
        # grab the *original* learning rate (so we can reset it later), and then set the *starting* learning rate
        orig_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, start_lr)
        # construct a callback that will be called at the end of each batch, enabling us to
        # increase our learning rate as training progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        # check to see if we are using a data iterator
        if use_gen:
            self.model.fit_generator(
                train_data,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback])
        # otherwise, our entire training data is already in memory
        else:
            # train our model using Keras' fit method
            self.model.fit(
                train_data[0], train_data[1],
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[callback],
                verbose=verbose)
        # restore the original model weights and learning rate
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, orig_lr)

    def plot_loss(self, skip_begin=10, skip_end=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]
        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)

