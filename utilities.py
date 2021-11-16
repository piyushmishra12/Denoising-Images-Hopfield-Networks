import numpy as np
import matplotlib.pyplot as plt
import skimage.data as sd
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_mean

def preprocess(image, width=128, height=128):
    image = resize(image, (width, height))
    # converting to binary
    bin_image = image > threshold_mean(image)
    bin_image = 2*(bin_image*1) - 1
    
    # vectorising the image
    vector = np.reshape(bin_image,
                        (width * height))
    return vector

# function to get noisy/crappy data
def crappify(image_vector, level = 0.3):
	crappy_image = np.copy(image_vector)
	invert_pixel = np.random.binomial(n = 1,
		p = level,
		size = len(image_vector))
	for i, pixel in enumerate(image_vector):
		if invert_pixel[i]:
			crappy_image[i] = -1 * pixel
	return crappy_image

# converting back from a vector to a 2x2 matrix
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot_results(test, predicted):
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(test), 2)
    for i in range(len(test)):
        if i==0:
            axarr[i, 0].set_title("Input data")
            axarr[i, 1].set_title('Output data')
        axarr[i, 0].imshow(test[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(predicted[i])
        axarr[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_energies(energy_curves):
	n = len(energy_curves)
	for i in range(n):
		plt.subplot(n, 1, i+1)
		plt.plot(energy_curves[i])
	plt.xlabel("Iterations")
	plt.ylabel("Energy")
	plt.tight_layout()
	plt.show()

def get_images():
    camera = sd.camera()
    astronaut = rgb2gray(sd.astronaut())
    horse = sd.horse()
    chelsea = rgb2gray(sd.chelsea())
    return [camera, astronaut, horse, chelsea]
