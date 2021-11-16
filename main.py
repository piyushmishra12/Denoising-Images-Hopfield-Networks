# -*- coding: utf-8 -*-
import utilities
from hopfield import Hopfield
from hopfieldnetwork import HopfieldNetwork

def hopfield_from_scratch(asyn = False):
    images = utilities.get_images()
    preprocessed_images = [utilities.preprocess(img) for img in images]
    model = Hopfield(asyn=asyn)
    # assuming synchronous update
    # for asynchronous update, add "asyn=True" in hopfield.Hopfield
    W = model.train(preprocessed_images)
    # testing on crappy images
    test_images = [utilities.crappify(vector)
                   for vector in preprocessed_images]
    predicted, energy_curves = model.predict(test_images)
    utilities.plot_energies(energy_curves)
    utilities.plot_results(test_images, predicted)
    model.plot_weights()



if __name__ == '__main__':
    print("Asynchronously Updated Hopfield Network:")
    hopfield_from_scratch(True)

    print("Synchronously Updated Hopfield Network:")
    hopfield_from_scratch()


