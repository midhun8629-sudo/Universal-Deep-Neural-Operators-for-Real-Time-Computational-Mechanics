# Deep Operator Learning for Viscoelastic Constitutive Modeling (Neural UMAT)

## Overview
This repository contains a PyTorch-based implementation of deep operator learning techniques to model viscoelastic material behavior. Specifically, it trains a neural network to act as a **Neural UMAT (User Material)** surrogate, learning the complex mapping between continuous strain histories and stress responses governed by the Maxwell material model. 

By replacing classical explicit numerical integration with a trained neural surrogate, this approach significantly accelerates Finite Element Method (FEM) simulations while maintaining high physical fidelity.

## Key Features

* **Physics-Based Data Generation:** Custom simulator for the Maxwell viscoelastic model to generate high-fidelity ground truth data based on explicit time-stepping.
* **Universal Training Dataset:** A robust data generation pipeline that simulates diverse and arbitrary loading paths to ensure model generalization:
  * Standard Sine Waves
  * Ramp-Up Sine Waves (typical in FEM start-up phases)
  * Random Walk / Noise (arbitrary loading)
  * Partial Masking (simulating varying initial conditions)
* **Operator Network Architectures:** * Implementation of a standard **DeepONet** (Branch and Trunk networks) for continuous operator learning.
  * Implementation of a deep **NeuralUMAT** (Multilayer Perceptron) optimized for discrete time-step strain-history mapping.
* **Direct FEM Benchmarking:** A built-in simulation loop that directly compares the classical numerical solver against the neural surrogate across hundreds of simulated elements to quantify speedup and error.

## Benchmark Results
The model was tested on a mock 1D FEM simulation loop evaluating 500 elements over multiple time steps using a challenging ramping sine wave load. 

* **Speedup:** **~22.2x faster** than the classical explicit integration method.
* **Accuracy:** **~3.89% Relative L2 Error** compared to the ground-truth physical equations.

*(Results are based on the standard `NeuralUMAT` architecture trained over 14,500 epochs).*

## Dependencies
To run the notebook, you need the following Python libraries installed:
* `torch` (PyTorch)
* `numpy`
* `matplotlib`

## Project Structure (Notebook Flow)

1. **Physics Generation:** Defines the material properties ($E = 10.0$, $\tau = 2.0$) and generates normalized strain-to-stress datasets using the Maxwell governing equations.
2. **DeepONet Implementation:** Trains a continuous operator network separating the input functional space (strain history sensors) and the evaluation domain (time).
3. **Universal Neural UMAT:** Generates a broader, multi-modal dataset to train a deeper Sequential MLP, capable of handling arbitrary loading envelopes.
4. **Evaluation & Visualization:** Runs the side-by-side benchmark, plotting the stress response overlay and a bar chart detailing the computational time reduction.

## Usage
Simply open `Operator_Learning_ (2).ipynb` in Jupyter Notebook or Google Colab and run the cells sequentially. The code will automatically generate the synthetic dataset, train the models, and output the comparative benchmark plots. 

No external datasets are required as all physical data is generated programmatically within the notebook.
