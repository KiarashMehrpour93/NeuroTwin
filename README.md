````markdown
# NeuroTwin

### An AI-Based Personalized Computational Model for Adaptive Neurorehabilitation

> **Model the Brain. Simulate the Future. Personalize Recovery.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE-EEG%20Processing-orange.svg)](https://mne.tools/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple.svg)]()
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)]()

**Version:** NeuroTwin 3.0  
**Status:** Research Prototype  
**Author:** Kiarash Mehrpour  
**Year:** 2026

---

## Overview

NeuroTwin is an AI-driven computational neuroscience research prototype designed to explore how EEG-derived information, machine learning, functional connectivity, brain-network analysis, neural dynamics, and adaptive computational modeling can be combined into a personalized computational model of brain state.

The central concept is a **Digital Brain Twin**: a simplified computational representation of an individual's measured neural state that can be used to simulate hypothetical computational perturbations and compare their modeled responses.

NeuroTwin is not intended to replace the human brain with a complete biological simulation.

Instead, it provides a computational framework for:

- EEG preprocessing
- Neural feature extraction
- Spectral analysis
- CSP-based feature extraction
- Machine-learning baselines
- Functional connectivity analysis
- Brain-network construction
- Graph-theoretic analysis
- Personalized computational profiles
- Neural-dynamics simulation
- Computational perturbation strategies
- Multi-stage simulation
- Robustness evaluation
- Sensitivity analysis
- Adaptive computational modeling
- Cross-subject computational generalization

The long-term research direction is to investigate whether increasingly personalized computational models can become more informative as additional longitudinal and multimodal neural data become available.

---

## Research Motivation

Human neural activity is highly complex, dynamic, and individual-specific.

Two individuals can exhibit different neural patterns even when performing similar tasks.

This creates an important computational challenge:

> Can neural measurements be transformed into an individualized computational representation that can be used to simulate hypothetical changes in neural dynamics?

NeuroTwin explores this question through a pipeline that combines:

```text
EEG
 │
 ▼
Preprocessing
 │
 ▼
Feature Extraction
 │
 ├── Spectral Features
 ├── Frequency Bands
 └── CSP Features
 │
 ▼
Functional Connectivity
 │
 ├── PLV
 └── Coherence
 │
 ▼
Brain Network
 │
 ▼
Personalized Computational Profile
 │
 ▼
Digital Brain Twin
 │
 ▼
Neural Dynamics
 │
 ▼
Computational Perturbation
 │
 ▼
Simulation
 │
 ▼
Robust Evaluation
 │
 ▼
Adaptive Computational Modeling
```

The system therefore treats the brain as a measurable and computationally representable dynamical system rather than attempting to claim a complete biological simulation.

---

## Core Concept

The NeuroTwin architecture can be summarized as:

```text
Human Brain
     │
     ▼
EEG Measurements
     │
     ▼
Signal Processing
     │
     ▼
Neural Features
     │
     ▼
Functional Connectivity
     │
     ▼
Brain Network
     │
     ▼
Personalized Computational Profile
     │
     ▼
Digital Brain Twin
     │
     ▼
Neural Dynamics
     │
     ▼
Computational Simulation
     │
     ├──────────────┬──────────────┐
     ▼              ▼              ▼
Strategy A       Strategy B      Strategy C
     │              │              │
     └──────────────┼──────────────┘
                    ▼
            Robust Evaluation
                    │
                    ▼
          Adaptive Computational Model
                    │
                    ▼
        Updated Computational State
```

---

## Scientific Positioning

NeuroTwin is designed as a **computational neuroscience and AI research prototype**.

It does not claim that its simulations represent the complete biological mechanisms of the human brain.

The project focuses on measurable computational representations derived from EEG and mathematical models of neural dynamics.

### Important terminology

NeuroTwin uses:

- **Computational perturbation strategies**
- **Model-Selected Strategy**
- **Cross-Subject Computational Generalization**
- **Digital Brain Twin**
- **Functional Connectivity**
- **Neural Dynamics**

These terms are intentionally used instead of making direct clinical claims.

For example:

> Strategy C is the highest-scoring strategy according to the current computational model.

It should **not** be described as:

> Strategy C is the best medical treatment.

---

# System Architecture

## Project Structure

```text
NeuroTwin/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── models/
│   └── baseline_model.joblib
│
├── results/
│   ├── metrics.json
│   ├── v2_metrics.json
│   ├── connectivity/
│   └── generalization/
│
└── src/
    ├── __init__.py
    ├── adaptive_model.py
    ├── csp_features.py
    ├── download_data.py
    ├── evaluation.py
    ├── features.py
    ├── neurotwin.py
    ├── preprocessing.py
    └── train.py
```

---

## Component Responsibilities

### `app.py`

Main Streamlit research dashboard.

Responsibilities include:

- Subject selection
- Simulation configuration
- Experiment configuration
- Visualization
- Experiment execution
- Strategy comparison
- Results presentation
- Research-oriented dashboard controls

---

### `src/preprocessing.py`

Responsible for EEG preprocessing.

Main concepts include:

- EEG loading
- Channel handling
- Referencing
- Filtering
- Event handling
- Epoch preparation
- Numerical conversion

---

### `src/features.py`

Responsible for feature engineering.

Main components include:

- Power Spectral Density
- Welch spectral estimation
- Frequency bands
- Log power
- Feature vectors
- Subject-level representations

---

### `src/csp_features.py`

Provides CSP-based feature extraction.

CSP is used to transform multichannel EEG into spatially discriminative representations.

---

### `src/neurotwin.py`

Contains the core Digital Brain Twin and neural-dynamics abstraction.

Responsibilities include:

- Computational state representation
- Connectivity/interaction representation
- Neural state evolution
- Stability mechanisms
- Computational simulation

---

### `src/adaptive_model.py`

Implements adaptive computational modeling.

The model can update its computational state using information generated by the simulation process.

The current system does **not** use real post-intervention EEG as an adaptive feedback signal.

---

### `src/evaluation.py`

Responsible for experimental evaluation.

Includes concepts such as:

- Repeated experiments
- Multiple random seeds
- Confidence intervals
- Sensitivity analysis
- Cross-subject evaluation
- Robustness metrics

---

### `src/train.py`

Responsible for training machine-learning models.

---

### `src/download_data.py`

Responsible for dataset acquisition and preparation.

---

# Dataset

NeuroTwin uses the:

## PhysioNet EEG Motor Movement/Imagery Database

**EEGMMIDB**

Dataset:

https://physionet.org/content/eegmmidb/1.0.0/

The current experimental configuration uses:

- 10 subjects
- 30 EDF files
- Runs R04, R08, and R12
- T0/T1/T2 annotations where applicable

Raw EEG files are intentionally not included in this repository.

---

## Data Policy

Raw EEG files should not be committed to the Git repository.

The following data formats are excluded from version control:

```text
*.edf
*.fif
*.bdf
*.set
*.cnt
```

The raw data directory is also ignored.

This keeps the repository lightweight and avoids unnecessarily redistributing the original dataset.

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| MNE | EEG processing |
| NumPy | Numerical computation |
| SciPy | Scientific computing |
| Pandas | Data processing |
| scikit-learn | Machine learning |
| NetworkX | Brain-network analysis |
| Matplotlib | Visualization |
| Streamlit | Research dashboard |
| Joblib | Model persistence |

---

# EEG Processing Pipeline

The EEG processing pipeline follows the general structure:

```text
Raw EEG
  │
  ▼
Channel Handling
  │
  ▼
Reference
  │
  ▼
Filtering
  │
  ▼
Event Extraction
  │
  ▼
Epoch Preparation
  │
  ▼
Numerical Representation
  │
  ▼
Feature Extraction
```

The goal is to transform raw neural recordings into structured numerical representations suitable for machine-learning and computational modeling.

---

# Feature Engineering

NeuroTwin extracts multiple classes of EEG-derived information.

## Power Spectral Density

Power Spectral Density is used to characterize the distribution of signal power across frequencies.

---

## Welch Spectral Estimation

Welch's method provides a stable spectral estimate by dividing the signal into overlapping segments and averaging their periodograms.

---

## Frequency Bands

The system works with frequency-band representations such as:

```text
Delta
Theta
Alpha
Beta
Gamma
```

These features can be transformed into numerical representations for machine-learning and computational modeling.

---

## Log Power

Log-transformed spectral power can be used to reduce scale differences and provide more stable numerical representations.

---

## Subject-Level Representation

The feature pipeline produces a subject-level computational representation that can later be combined with connectivity and network information.

---

# Machine Learning Baseline

The initial baseline uses:

```text
StandardScaler
      │
      ▼
LogisticRegression
```

Configuration:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
```

The baseline provides a reference point for evaluating more advanced representations.

---

# Baseline Results

## Logistic Regression

Current recorded result:

| Metric | Result |
|---|---:|
| Accuracy | 0.5000 |

Confusion matrix:

```text
[[27, 20],
 [25, 18]]
```

The baseline result is intentionally reported without exaggeration.

It demonstrates that a simple feature-based classifier alone does not provide strong predictive performance in the current experimental configuration.

---

# CSP + LDA

A second computational pipeline combines:

```text
EEG
 │
 ▼
CSP
 │
 ▼
LDA
 │
 ▼
Classification
```

Current recorded results:

| Metric | Mean |
|---|---:|
| Accuracy | 0.5356 |
| F1 | 0.4769 |

Confusion matrix:

```text
[[125, 105],
 [104, 116]]
```

These results are experimental results from the current research prototype and should not be interpreted as clinical performance.

---

# Functional Connectivity

NeuroTwin includes functional connectivity analysis using:

- Phase Locking Value (PLV)
- Coherence

The current connectivity configuration focuses on:

```text
8–30 Hz
```

A combined connectivity representation can be calculated from:

```text
Combined Connectivity
=
0.5 × PLV
+
0.5 × Coherence
```

Connectivity matrices can then be transformed into brain-network representations.

---

## Important Scientific Note

Functional connectivity represents a statistical relationship between neural signals.

It should **not** automatically be interpreted as:

- anatomical connectivity
- direct physical connections
- causal connectivity

Therefore, NeuroTwin describes these measurements as **functional/statistical relationships**.

---

# Brain Network Analysis

The connectivity matrix can be represented as a weighted graph.

```text
Connectivity Matrix
        │
        ▼
      Graph
        │
   ┌────┼────┐
   ▼    ▼    ▼
 Nodes Edges Weights
        │
        ▼
 Graph Metrics
```

NetworkX is used for graph analysis.

Metrics include:

- Degree
- Strength
- Clustering
- Betweenness
- Density
- Mean degree
- Mean strength
- Mean clustering
- Mean betweenness

For weighted shortest-path calculations, stronger connectivity can correspond to shorter graph distances.

---

# Personalized Computational Profile

The Digital Brain Twin is built from multiple layers of information.

```text
Personalized Computational Profile
│
├── EEG Features
│
├── Spectral Features
│
├── Connectivity
│
├── Network Metrics
│
├── Baseline State
│
└── Model Parameters
```

The objective is to represent each subject using a structured computational state rather than relying exclusively on a generic population-level model.

---

# Digital Brain Twin

The NeuroTwin Digital Brain Twin is a **computational abstraction** of measured neural characteristics.

It is not intended to reproduce every biological process occurring in the human brain.

The Digital Brain Twin represents:

```text
Observed Measurements
        │
        ▼
Numerical Features
        │
        ▼
Connectivity
        │
        ▼
Network Representation
        │
        ▼
Computational State
        │
        ▼
Dynamic Simulation
```

This abstraction allows hypothetical computational experiments to be performed without claiming that the simulation is a complete biological model.

---

# Neural Dynamics

The core neural-dynamics abstraction follows a recurrent state-update equation:

```text
x(t+1) =
tanh(
    (1 - damping) × x(t)
    +
    coupling × W × x(t)
    +
    input
)
```

Where:

| Variable | Meaning |
|---|---|
| `x(t)` | Current computational neural state |
| `damping` | State decay factor |
| `coupling` | Interaction strength |
| `W` | Connectivity/interaction matrix |
| `input` | External computational input |
| `tanh` | Nonlinear activation |

This is a mathematical abstraction of neural dynamics.

It is **not** a complete biological simulation of the human brain.

---

# Stability Control

Dynamic systems can become unstable when recurrent interactions become too strong.

NeuroTwin therefore uses computational stability mechanisms such as:

- Row normalization
- Spectral-radius control
- State clipping
- Coupling control

The current computational configuration targets a spectral radius of approximately:

```text
0.95
```

The objective is to keep simulated states numerically stable during repeated updates.

---

# Computational Perturbation Strategies

NeuroTwin evaluates multiple hypothetical computational strategies.

The current strategies are:

```text
Strategy A
Strategy B
Strategy C
```

These are **computational perturbation strategies**.

They are not medical treatments.

The system compares their simulated responses using multiple computational metrics.

---

# Strategy Evaluation

Current experimental results:

| Metric | Strategy A | Strategy B | Strategy C |
|---|---:|---:|---:|
| Similarity | 0.9260 | 0.9356 | **0.9572** |
| Stability | 0.9858 | 0.9878 | **0.9920** |
| Feature Shift | 0.0048 | 0.0041 | **0.0027** |
| Response Std | 0.0006 | 0.0005 | 0.0016 |
| CI95 | 0.0002 | 0.0001 | 0.0004 |
| Pattern Preservation | 0.9930 | 0.9949 | **0.9984** |
| Compatibility | 0.9456 | 0.9529 | **0.9690** |
| Consistency | 0.9994 | 0.9995 | 0.9984 |
| Uncertainty | 0.0006 | 0.0005 | 0.0016 |
| Adaptation Potential | 0.9779 | 0.9809 | **0.9866** |
| Adjusted | 0.9453 | 0.9527 | **0.9683** |

Under the current computational scoring system, **Strategy C receives the highest overall computational scores**.

This does not mean Strategy C is the best rehabilitation treatment or a clinically superior intervention.

It means that Strategy C produced the strongest result according to the current mathematical simulation and evaluation framework.

---

# Multi-Stage Simulation

NeuroTwin supports multi-stage computational simulations.

Conceptually:

```text
Stage 1
   │
   ▼
State 1
   │
   ▼
Stage 2
   │
   ▼
State 2
   │
   ▼
Stage 3
   │
   ▼
...
   │
   ▼
Final Computational State
```

A model-selected strategy can be selected at each computational stage.

This allows the system to investigate whether sequential perturbations produce different computational trajectories compared with a single-step simulation.

---

# Model-Selected Strategy

The system can compare candidate strategies and select a strategy according to the computational evaluation metrics.

Conceptually:

```text
Candidate Strategies
        │
        ▼
Simulation
        │
        ▼
Evaluation Metrics
        │
        ▼
Computational Ranking
        │
        ▼
Model-Selected Strategy
```

The selected strategy is a **model output**.

It is not a clinical recommendation.

---

# Multi-Seed Robustness

To reduce dependence on a single random initialization, NeuroTwin supports multiple experiment seeds.

Current seed configuration:

```text
1
2
3
4
5
6
7
8
9
10
```

The system can calculate:

- Mean
- Standard deviation
- Minimum
- Maximum
- 95% confidence interval
- Consistency
- Uncertainty

A conceptual experiment is:

```text
Seed 1  ─┐
Seed 2  ─┤
Seed 3  ─┤
Seed 4  ─┤
Seed 5  ─┤
Seed 6  ─┤
Seed 7  ─┤
Seed 8  ─┤
Seed 9  ─┤
Seed 10 ─┘
          │
          ▼
   Robustness Summary
```

---

# Sensitivity Analysis

NeuroTwin supports computational sensitivity analysis across different perturbation intensities.

Example intensity range:

```text
0.1
0.2
0.3
...
1.0
```

The purpose is to examine how model behavior changes as computational parameters vary.

Sensitivity analysis can reveal:

- Stable regions
- High-response regions
- Parameter sensitivity
- Instability regions
- Robust parameter ranges

These findings are computational observations rather than biological or clinical conclusions.

---

# Adaptive Computational Modeling

One of the central concepts of NeuroTwin is adaptive computational modeling.

The conceptual loop is:

```text
Initial Computational Profile
            │
            ▼
       Simulation
            │
            ▼
 Computational Response
            │
            ▼
      Profile Update
            │
            ▼
 Updated Computational State
            │
            ▼
     Next Simulation
            │
            └───────────────┐
                            ▼
                     Adaptive Loop
```

The current adaptive mechanism uses information generated by the computational model.

It does **not** currently receive real post-intervention EEG measurements as feedback.

Therefore, the current system should be described as:

> Adaptive Computational Modeling

rather than claiming real-world adaptive neurorehabilitation.

---

# Cross-Subject Computational Generalization

NeuroTwin evaluates computational behavior across multiple subjects.

Current configuration:

```text
Subjects = 10
```

The objective is to investigate whether computational representations and simulation behavior remain consistent across different individuals.

This is referred to as:

> **Cross-Subject Computational Generalization**

This terminology is intentionally preferred over claims of clinical generalization.

The current experiment does not establish that the system generalizes clinically to the broader human population.

---

# Research Dashboard

NeuroTwin provides a Streamlit-based research dashboard.

Main controls include:

| Parameter | Current Range |
|---|---|
| Subject ID | 1–10 |
| Stages | 1–7 |
| Repetitions | 10–100 |
| Experiment Seeds | 1–10 |
| Main Seed | 42 |
| Intensity | 0.1–1.0 |
| Connectivity Threshold | 0.05–0.90 |
| Neural Dynamics Steps | 20–300 |
| Coupling | 0.01–0.50 |

The dashboard is designed to make computational experiments easier to configure, execute, visualize, and compare.

---

# Feature Matrix

| Capability | Status |
|---|---|
| EEG Processing | ✅ |
| EEG Feature Extraction | ✅ |
| PSD Analysis | ✅ |
| Welch Spectral Estimation | ✅ |
| Frequency Bands | ✅ |
| Logistic Regression Baseline | ✅ |
| CSP | ✅ |
| LDA | ✅ |
| Functional Connectivity | ✅ |
| PLV | ✅ |
| Coherence | ✅ |
| Brain Networks | ✅ |
| Graph Metrics | ✅ |
| Personalized Computational Profile | ✅ |
| Digital Brain Twin Abstraction | ✅ |
| Neural Dynamics | ✅ |
| Stability Control | ✅ |
| Computational Perturbation Strategies | ✅ |
| Multi-Stage Simulation | ✅ |
| Multi-Seed Robustness | ✅ |
| Sensitivity Analysis | ✅ |
| Adaptive Computational Modeling | ✅ |
| Cross-Subject Computational Generalization | ✅ |
| Streamlit Dashboard | ✅ |
| Longitudinal Real Intervention Data | ❌ |
| Clinical Validation | ❌ |
| Clinical Trial Validation | ❌ |
| Regulatory Validation | ❌ |
| Full Biological Brain Simulation | ❌ |

---

# Understanding the Current Limitations

The ❌ items in the feature matrix are intentional research boundaries.

## Longitudinal Real Intervention Data

This would require real neural measurements from individuals across multiple timepoints, for example:

```text
Before Intervention
        │
        ▼
Intervention
        │
        ▼
After Intervention
        │
        ▼
New EEG
        │
        ▼
Longitudinal Comparison
```

NeuroTwin does not currently have such a dataset.

---

## Clinical Validation

Clinical validation would require independent validation using clinically relevant populations, protocols, outcomes, and study designs.

The current computational experiments do not constitute clinical validation.

---

## Clinical Trial Validation

A clinical trial would require an appropriately designed prospective human study.

NeuroTwin has not performed a clinical trial.

---

## Regulatory Validation

A medical product intended for clinical use can require regulatory review depending on jurisdiction and intended use.

NeuroTwin has not undergone regulatory validation or approval.

---

## Full Biological Brain Simulation

The human brain contains enormous biological complexity, including:

- neurons
- synapses
- neurotransmitters
- glial cells
- molecular mechanisms
- structural connectivity
- vascular systems
- cellular dynamics
- neurochemical processes

NeuroTwin does not simulate all of these biological mechanisms.

Its Digital Brain Twin is a simplified computational abstraction.

---

# Reproducibility

NeuroTwin is designed around reproducible computational experiments.

Important reproducibility factors include:

- Random seeds
- Experiment parameters
- Dataset version
- Subject selection
- Number of repetitions
- Model configuration
- Result export

The primary seed used in the dashboard is:

```text
42
```

Additional robustness experiments use multiple seeds.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/KiarashMehrpour93/NeuroTwin.git
cd NeuroTwin
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running NeuroTwin

Launch the Streamlit dashboard:

```bash
streamlit run app.py --server.port 7654
```

Then open the local Streamlit address displayed in the terminal.

---

# Training

To train the baseline model:

```bash
python src/train.py
```

The trained model can be stored under:

```text
models/
```

---

# Dataset Preparation

Dataset acquisition and preparation are handled through:

```text
src/download_data.py
```

The repository does not include the raw EEG recordings.

Before using the dataset, users should review the original PhysioNet dataset terms and conditions.

---

# Results

Results are stored under:

```text
results/
├── metrics.json
├── v2_metrics.json
├── connectivity/
└── generalization/
```

These outputs are intended to support reproducibility and experiment comparison.

---

# Research Workflow

A complete NeuroTwin experiment can be represented as:

```text
Dataset
   │
   ▼
EEG Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Connectivity Analysis
   │
   ▼
Brain Network
   │
   ▼
Personalized Profile
   │
   ▼
Digital Brain Twin
   │
   ▼
Neural Dynamics
   │
   ▼
Computational Strategies
   │
   ▼
Multi-Stage Simulation
   │
   ▼
Multi-Seed Evaluation
   │
   ▼
Sensitivity Analysis
   │
   ▼
Adaptive Computational Modeling
   │
   ▼
Cross-Subject Analysis
```

---

# Experimental Philosophy

NeuroTwin follows several principles.

## No Clinical Overclaiming

Computational results should not automatically be converted into medical conclusions.

---

## Explicit Uncertainty

Variability, confidence intervals, standard deviations, and uncertainty are treated as important experimental outputs.

---

## Reproducibility

Experiments should be reproducible through explicit parameters, seeds, and documented datasets.

---

## Individualization

The system prioritizes subject-specific computational representations instead of assuming every individual has the same neural state.

---

## Computational Transparency

Model outputs should be understandable as mathematical/computational results rather than unexplained claims.

---

# Scientific Limitations

NeuroTwin has important limitations.

### Dataset limitations

The current experiments are based on a public EEG motor movement/imagery dataset rather than a longitudinal clinical neurorehabilitation dataset.

### Biological limitations

The Digital Brain Twin is a mathematical abstraction and does not reproduce complete biological brain mechanisms.

### Connectivity limitations

Functional connectivity measures statistical relationships and should not be interpreted automatically as anatomical or causal connections.

### Adaptation limitations

The current adaptive model does not receive real post-intervention EEG as feedback.

### Generalization limitations

Cross-subject experiments do not establish clinical generalization.

### Clinical limitations

The system has not undergone clinical validation, clinical trials, or regulatory validation.

---

# Future Research

The long-term research roadmap includes:

- Dynamic functional connectivity
- Directed connectivity
- Graph Neural Networks
- Temporal models
- Temporal Transformers
- Neural ODEs
- Bayesian personalization
- Longitudinal modeling
- Multimodal neural data
- Structural brain information
- Real post-intervention EEG
- Uncertainty quantification
- Larger datasets
- Independent external validation
- Prospective research

---

# Research Roadmap

```text
Current
EEG + ML + Connectivity + Brain Networks + Digital Twin
        │
        ▼
Dynamic Connectivity
        │
        ▼
Graph Neural Networks
        │
        ▼
Temporal Models
        │
        ▼
Longitudinal Personalization
        │
        ▼
Real Intervention Data
        │
        ▼
Independent Validation
        │
        ▼
Clinical Research
```

---

# Privacy and Data Governance

If future versions incorporate real clinical or longitudinal human data, additional requirements will become important.

These may include:

- Data anonymization
- Privacy protection
- Secure storage
- Access control
- Ethical review
- Data governance
- Consent procedures
- Responsible data sharing

The current public dataset workflow does not represent a complete clinical data-management framework.

---

# Responsible AI

NeuroTwin is intended as a research platform.

Responsible use requires avoiding unsupported conclusions about:

- diagnosis
- treatment
- rehabilitation outcomes
- patient prognosis
- medical superiority
- individual medical decisions

Computational simulations should remain clearly separated from validated clinical evidence.

---

# Scientific Disclaimer

NeuroTwin is a research prototype for computational neuroscience and artificial intelligence.

It is **not**:

- a medical device
- a diagnostic system
- a treatment system
- a rehabilitation prescription system
- a clinical decision-support system
- a validated medical intervention
- a clinical trial
- a regulatory-approved system

The computational perturbation strategies represent hypothetical simulations.

A Model-Selected Strategy is a model-generated computational result and is **not a clinical recommendation**.

Functional connectivity represents statistical relationships between signals and should not automatically be interpreted as anatomical or causal connectivity.

The Digital Brain Twin is a simplified computational abstraction and is not a complete biological simulation of the human brain.

Any future clinical application would require appropriate longitudinal data, independent validation, prospective clinical research, ethical oversight, and applicable regulatory processes.

---

# Citation

If you reference NeuroTwin in academic or research work, please cite:

```text
Mehrpour, K. (2026).
NeuroTwin: An AI-Based Personalized Computational Model for Adaptive Neurorehabilitation.
GitHub Research Project.
```

---

# Author

**Kiarash Mehrpour**

GitHub:

https://github.com/KiarashMehrpour93

Portfolio:

https://kiarash-1393.github.io/kiacoder/

---

# Project Identity

```text
NeuroTwin 3.0

An AI-Based Personalized Computational Model
for Adaptive Neurorehabilitation

Model the Brain.
Simulate the Future.
Personalize Recovery.

Status: Research Prototype
Author: Kiarash Mehrpour
Year: 2026
```

---

# License and Copyright

## All Rights Reserved

© 2026 Kiarash Mehrpour. All Rights Reserved.

NeuroTwin is an independent research project developed by Kiarash Mehrpour.

The source code of this repository is publicly available for viewing and research evaluation purposes.

Public availability of this repository does **not** grant permission to copy, modify, redistribute, sublicense, publish, commercially use, or create derivative works from the NeuroTwin source code or substantial portions of it without explicit written permission from the author.

### Permissions

| Activity | Permission |
|---|---|
| View the source code | ✅ Allowed |
| View the project for research evaluation | ✅ Allowed |
| Reference the project academically | ✅ Allowed |
| Cite the project | ✅ Allowed |
| Copy source code | ❌ Not allowed without permission |
| Modify source code | ❌ Not allowed without permission |
| Redistribute source code | ❌ Not allowed without permission |
| Create derivative works | ❌ Not allowed without permission |
| Commercial use | ❌ Not allowed without permission |
| Republish substantial portions | ❌ Not allowed without permission |
| Sublicense the original code | ❌ Not allowed |

### Academic and Research References

Researchers, students, reviewers, and educational users may reference and cite the project.

Citation:

```text
Mehrpour, K. (2026).
NeuroTwin: An AI-Based Personalized Computational Model for Adaptive Neurorehabilitation.
GitHub Research Project.
```

Citation of the project does not grant permission to reproduce or redistribute its source code.

### Third-Party Components

NeuroTwin may use third-party libraries, frameworks, datasets, models, and other resources.

Those components remain subject to their respective licenses and terms.

This copyright statement applies to the original NeuroTwin code and original project materials created by Kiarash Mehrpour.

Third-party software and datasets are not relicensed by this notice.

### Permission Requests

Requests for permission to reproduce, modify, redistribute, publish, or commercially use NeuroTwin source code should be directed to the author.

---

# Final Vision

The long-term vision of NeuroTwin is to explore a computational pipeline in which neural measurements can be transformed into increasingly personalized computational representations.

```text
                    HUMAN BRAIN
                         │
                        EEG
                         │
                         ▼
                Neural Measurements
                         │
                         ▼
                Computational Model
                         │
                         ▼
                  DIGITAL TWIN
                         │
                         ▼
                  Neural Dynamics
                         │
                         ▼
               Computational Simulation
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         Strategy A  Strategy B  Strategy C
             │           │           │
             └───────────┼───────────┘
                         ▼
                 Robust Evaluation
                         │
                         ▼
                 Adaptive Modeling
                         │
                         ▼
             Updated Computational State
```

> **Model the Brain. Simulate the Future. Personalize Recovery.**
````
