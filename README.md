# GravSIRENSH

Repository for hybrid implicit neural representations of the gravity field that combine **Spherical Harmonic (SH) basis functions** with **SIREN networks** (Sitzmann et al., 2020).

## Overview

The purpose of this repository is to train **hybrid** and **numerical** gravity field models.

- The **numerical model** follows the approach of Martin and Schaub (2022).
- It uses the standard mean squared error loss:

\[
Loss = \frac{1}{N}\sum_{i=1}^{N} \left|x_n - \hat{x}_n\right|^2
\]

where  
- $\hat{x}$ represents the **predicted value**,  
- $x$ represents the **true value**,  

and the target variable can correspond to either **gravitational potential** or **acceleration**.

## Repository Contents

This repository includes:

- **PyTorch implementations** of the hybrid and numerical models  
- **Spherical Harmonic basis functions** implemented using `pyshtools`  
- **Data generators** for training datasets  
- **Logging and visualization tools** for model training and evaluation  

## Example

The file **`Examples.ipynb`** provides a minimal working example that:

1. Trains a **hybrid gravity field model**
2. Trains a **numerical gravity field model**
3. Visualizes the spatial distribution of the results

## Status

⚠️ This repository is currently intended **for research purposes only** and is **not designed for production use**.
