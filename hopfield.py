#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:58:41 2021

@author: piyushmishra
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm


class Hopfield(object):
	def __init__(self, asyn=False):
		self.asyn = asyn
		
	def train(self, data):
		print("Training:")
		self.neurons = data[0].shape[0]
		n = len(data)

		# initialising the weights for the Hebb's rule
		W = np.zeros((self.neurons, self.neurons))
		rho = np.sum([np.sum(t) for t in data]) / (n * self.neurons)

		for i in tqdm(range(n)):
			t = data[i] - rho
			W += np.outer(t, t)

		diagW = np.diag(np.diag(W))
		W = W - diagW
		W = W / n

		self.W = W
		return self.W

	def calculate_energy(self, s):
		e = -0.5 * s @ self.W @ s + np.sum(s * self.threshold)
		return e

	def compute(self, s0):
		if self.asyn == False:
			s = s0
			energy = self.calculate_energy(s)
			energies = [energy]  # to trace the history of energies
			for i in range(self.iterations):
				s = np.sign(self.W@s - self.threshold)
				next_energy = self.calculate_energy(s)
				energies.append(next_energy)
				if energy == next_energy:  # stopping criterion, no change in energy
					return s, energies
				energy = next_energy
			return s, energies
		else:  # Only one unit is updated at a time. This unit can be picked at random, or a pre-defined order can be imposed from the very beginning.
			s = s0
			energy = self.calculate_energy(s)
			energies = [energy]  # to trace the history of energies
			for i in range(self.iterations):
				for j in range(100):
					lucky_neuron = np.random.randint(0, self.neurons)
					s[lucky_neuron] = np.sign(self.W[lucky_neuron].T@s - self.threshold)
				next_energy = self.calculate_energy(s)
				energies.append(next_energy)
				if energy == next_energy:
					return s, energies
				energy = next_energy
			return s, energies

	def predict(self, test_data, iterations=20, threshold=0):
		print("Predicting:")
		self.iterations = iterations
		self.threshold = threshold

		# in order to stay away from call by reference
		running_data = np.copy(test_data)
		predicted = []
		energy_curves = []
		for i in tqdm(range(len(test_data))):
			s, energies = self.compute(running_data[i])
			predicted.append(s)
			energy_curves.append(energies)
		return predicted, energy_curves

	def plot_weights(self):
		w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
		plt.colorbar(w_mat)
		plt.title("Network Weights")
		plt.tight_layout()
		plt.show()