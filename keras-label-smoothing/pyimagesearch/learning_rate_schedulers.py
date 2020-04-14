# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
#   LEARNING RATE DECAY
# -----------------------------
class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # Compute the set of learning rates for each corresponding epoch
        lrs = [self(i) for i in epochs]
        # The Learning Rate Schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")


# -----------------------------
#   STEP DECAY
# -----------------------------
class StepDecay(LearningRateDecay):
    def __init__(self, init_alpha=0.01, factor=0.25, drop_every=10):
        # Store the base initial learning rate, drop factor and epochs to drop every decay
        self.init_alpha = init_alpha
        self.factor = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        # Compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.drop_every)
        alpha = self.init_alpha * (self.factor ** exp)
        # Return the learning rate
        return float(alpha)


# -----------------------------
#   POLYNOMIAL DECAY
# -----------------------------
class PolynomialDecay(LearningRateDecay):
    def __init__(self, max_epochs=100, init_alpha=0.01, power=1.0):
        # Store the maximum number of epochs, base learning rate and the power of the polynomial
        self.max_epochs = max_epochs
        self.init_alpha = init_alpha
        self.power = power

    def __call__(self, epoch):
        # Compute the new learning rate based on the polynomial decay
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        alpha = self.init_alpha * decay
        # Return the new learning rate
        return float(alpha)


# -----------------------------
#   MAIN FUNCTION
# -----------------------------
if __name__ == "__main__":
    # Plot a step-based decay which drops by a factor of 0.5 every 25 epochs
    sd = StepDecay(init_alpha=0.01, factor=0.5, drop_every=25)
    sd.plot(range(0, 100), title="Step-based Decay")
    plt.show()
    # Plot a linear decay by using a power of 1
    pd = PolynomialDecay(power=1)
    pd.plot(range(0, 100), title="Linear Decay (p=1)")
    plt.show()
    # Show a polynomial decay with a steeper drop by increasing the power value
    pd = PolynomialDecay(power=5)
    pd.plot(range(0, 100), title="Polynomial Decay (p=5)")
    plt.show()
