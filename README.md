# <img src="https://projects.ugr.es/relief-t1d/assets/img/project/Logo.png" width="85" alt="RELIEF-T1D Logo" align="center" /> &nbsp;&nbsp; RELIEF-T1D Core

**RELIEF-T1D: Reliable nocturnal glucose event prediction for T1D**

[![Website](https://img.shields.io/badge/Website-projects.ugr.es%2Frelief--t1d-005691?style=flat-square&logo=googlechrome&logoColor=white)](https://projects.ugr.es/relief-t1d/index.html)

---

### At a Glance

The `relief-t1d-core` repository provides the source code and data processing pipelines for Continuous Glucose Monitoring (CGM) time series within the **RELIEF-T1D** project. It contains reusable and packageable software logic designed to support reliable nocturnal glucose event prediction models.

> **Further Information:** Visit the [RELIEF-T1D Website](https://projects.ugr.es/relief-t1d/index.html) for detailed project scope, updates, and research publications.

---

### Repository Structure

```text
relief-t1d-core/
├── config/              # Configuration files (YAML) for data pipelines
├── scripts/             # Workflow execution scripts
├── src/                 # Core Python package source code
│   ├── evaluation/      # Metrics and evaluation utilities for model performance
│   ├── models/          # Core model architecture definitions and algorithm components
│   └── preprocessing/   # CGM time-series preprocessing and feature extraction
└── .env.example         # Example of environment variables for data directory configuration  
```

---

### Repository Breakdown

**`config/`**
Contains YAML configuration files used to parameterize data pipelines, model behavior, training routines, and execution settings without modifying code. It defines dataset metadata and partition strategies, experiment setup (such as predictive algorithms, loss functions, history lengths, prediction horizons, validation strategy), model training hyperparameters (such as batch size, max epochs, early stopping), and path mapping rules resolving environment variables.

**`scripts/`**
Contains executable scripts designed to automate core workflow execution without altering package source code. It includes dedicated tools for generating prediction windows from raw continuous glucose monitoring time series and for training glucose prediction models using the parameters specified in the configuration files.

**`src/`**
Houses the core Python package logic organized into functional submodules. It incorporates `evaluation/` for computing clinical evaluation metrics tailored to T1D prediction, `models/` for implementing predictive architectures and custom loss functions designed to penalize clinical risk, and `preprocessing/` for feature extraction routines, such as generating time-series windows and cross-validation folds.

**`.env.example`**
Serves as an environment configuration template to be duplicated as .env in local setups. It allows defining local data directory paths, decoupling system-specific paths from execution scripts and preventing hardcoded file locations within the codebase.

---

### Other Code & Repositories

<table width="100%">
  <tr>
    <td width="50%" valign="top">
      <h4><a href="https://github.com/relief-t1d"><code>relief-t1d</code></a></h4>
      <p>Central repository of the <b>RELIEF-T1D</b> project outlining the framework and general structure of the code and repositories.</p>
      <p><sub><code>Project Scope</code> &nbsp;•&nbsp; <code>Project Framework</code></sub></p>
    </td>
    <td width="50%" valign="top">
      <h4><a href="https://github.com/relief-t1d/relief-t1d-data"><code>relief-t1d-data</code></a></h4>
      <p>Repository for storing, documenting, and versioning continuous glucose monitoring datasets used throughout the research.</p>
      <p><sub><code>CGM Datasets</code> &nbsp;•&nbsp; <code>Data Versioning</code> &nbsp;•&nbsp; <code>Documentation</code></sub></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h4><a href="https://github.com/relief-t1d/relief-t1d-experiments"><code>relief-t1d-experiments</code></a></h4>
      <p>Experimental workflows, scripts, and documented results providing all necessary information for full research reproducibility.</p>
      <p><sub><code>Reproducibility</code> &nbsp;•&nbsp; <code>Experiment Tracking</code> &nbsp;•&nbsp; <code>Research Logs</code></sub></p>
    </td>
    <td width="50%" valign="top">
      <h4><a href="https://github.com/relief-t1d/relief-t1d-models"><code>relief-t1d-models</code></a></h4>
      <p>Model registry housing stable, versioned artifacts and checkpoints of trained machine learning and deep learning models.</p>
      <p><sub><code>Model Registry</code> &nbsp;•&nbsp; <code>Trained Models</code> &nbsp;•&nbsp; <code>ML Artifacts</code></sub></p>
    </td>
  </tr>
</table>

---

### Acknowledgment

Grant PID2023-148188OA-I00 funded by: MICIU/AEI /10.13039/501100011033 and by ERDF, EU.

<img src="https://projects.ugr.es/relief-t1d/assets/img/project/MICIU+Cofinanciado+AEI.jpg" alt="Institutional Funding Logos" width="400" />
