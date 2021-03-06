# Denoising-Images-Hopfield-Networks
Collaborators: Piyush Mishra, Clémence Fournié

This repository contains the code for the methodology to remove noise from images, as part of the final project titled "Comparative Study of Synchronous and Asynchronous Hopfield Networks for Image Denoising" for the course of Programming and Algorithms, taken by Julien Lefèvre and Jérémie Perrin. The aim of the project is to compare and contrast the working of synchronous and asynchronous Hopfield Networks and gauge their abilities in removing noise from a given image. This is a rudimentary project that works on binary images. The Hopfield network pipeline is created from scratch.

## Running the Program
1. Clone or download the zip of the repository into your machine.
2. `cd` into the repository, `cd path/to/Denoising-Images-Hopfield-Networks`. Note: Replace `path/to` with the actual path in your machine.
3. *Optional*: Make a virtual environment and activate it:`python -m venv env` then, `source env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the main file: `python main.py`

## Modules
We have created three modules for this project.
* The main module, `main.py` contains the skeleton code of the entire project. Here, one can find how the project is assembled in its entirety and how the flow of control passes from one function of the project to another.
* The utilities module, `utilities.py` contains a set of "helper functions" which indeed help in carrying out important, yet regulatory tasks for the project.
* The hopfield module, `hopfield.py` contains the functions necessary for carrying out the Hopfield computations as member functions of the parent class Hopfield.

