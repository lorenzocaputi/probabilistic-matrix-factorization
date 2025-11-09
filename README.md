# Probabilistic Matrix Factorization

A research project implementing and analyzing probabilistic matrix factorization techniques.

## Overview

This repository contains research code and experiments related to probabilistic matrix factorization (PMF), including implementations of various algorithms and empirical studies.

## Project Structure

- `experiments/` - Jupyter notebooks and Python scripts for experiments
  - `diagonal_prior_bad_pairing_saddle_experiment.ipynb` - Diagonal prior analysis
  - `lemma9.py` - Implementation of Lemma 9
  - `map_landscape_experiments.ipynb` - MAP landscape analysis
  - `pmf_als_experiment_min.ipynb` - Minimal ALS experiment
  - `pmf_als_experiment_simplified.ipynb` - Simplified ALS experiment
- `latex_document/` - LaTeX documentation and research paper
  - `pmf.tex` - Main LaTeX document
  - `pmf.pdf` - Compiled PDF

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lorenzocaputi/probabilistic-matrix-factorization.git
cd probabilistic-matrix-factorization
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy >= 1.24
- scipy >= 1.10
- pandas >= 1.5
- matplotlib >= 3.7
- jax
- mpmath

## Usage

Launch Jupyter notebook to explore the experiments:
```bash
jupyter notebook experiments/
```

## Research

This project investigates various aspects of probabilistic matrix factorization, including algorithmic implementations, convergence analysis, and landscape optimization studies.
