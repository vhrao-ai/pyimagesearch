# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
#   FUNCTIONS
# -----------------------------
def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # Grab the dimensions of the image and use the dimensions to derive the center (x,y) coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # Compute the FFT to find the frequency transform, then shift the zero frequency component (i.e, DC component
    # located at the top-left corner) to the center where it will be more easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    # Check to see if the output needs to be visualized
    if vis:
        # Compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))
        # Display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # Display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # Show the plots
        plt.show()
    # Zero-out the center of the FFT shift (i.e., remove low frequencies), apply the inverse shift such that
    # the DC component once again becomes the top-left and then apply the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    # Compute the magnitude spectrum of the reconstructed image, then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # The image will be considered "blurry" if the mean value of the magnitudes is less than the threshold value
    return mean, mean <= thresh
