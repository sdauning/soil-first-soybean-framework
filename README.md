# Soil-first framework reveals dominant constraints closing global soybean yield gaps

## 1. Overview

This repository supports the study:

**“A soil-first framework reveals dominant constraints closing global soybean yield gaps”**

The study develops a soil-first machine learning framework to quantify the relative importance of soil, climate, and management factors in explaining global soybean yield variability. A Random Forest model combined with SHAP interpretation is used to identify dominant environmental constraints and yield gap drivers.

The analysis is based on global gridded datasets and aims to reveal how soil properties regulate soybean yield potential under varying climatic conditions.

---

## 2. Repository structure
code/
data_preprocessing.py 
RF_model.py 
SHAP_analysis.py 

processed_dataset.csv 
Variable_Dictionary.xlsx
requirements.txt 
LICENSE 
---

## 3. Data description

### 3.1 Processed dataset

The file `processed_dataset.csv` contains the final dataset used for model training and validation.

- Spatial scale: Global gridded dataset
- Temporal coverage: 2003–2016 (excluding 2008, 2011, and 2014 for independent validation)
- Unit of yield: t/ha
- Data type: Derived dataset (not raw observations)

To reduce interannual variability and improve robustness of yield–environment relationships, predictor variables were constructed using temporally averaged values across available years.

Independent validation was performed using the excluded years (2008, 2011, and 2014) to ensure temporal generalization of the model.

---

## 4. Variable description

All variables used in this study are defined in `Variable_Dictionary.xlsx`.

The dataset includes:

- Climate variables (temperature, precipitation, radiation, CO₂)
- Soil variables (pH, SOC, sand, silt, clay, CEC, bulk density)
- Geographic variables (longitude, latitude)
- Management-related variables (irrigation, where available)

Each variable is described with:

- Full name
- Unit
- Data source
- Ecological interpretation

---

## 5. Methods

### 5.1 Machine learning model

A Random Forest regression model was used to estimate global soybean yield based on environmental predictors.

- Algorithm: Random Forest Regression
- Implementation: scikit-learn
- Purpose: nonlinear mapping of environment–yield relationships

### 5.2 Soil-first framework

This study adopts a soil-first analytical framework, prioritizing soil properties as primary constraints in explaining yield variability, followed by climate and management factors.

### 5.3 Model interpretation

SHAP (SHapley Additive exPlanations) was used to:

- Quantify feature importance
- Identify nonlinear responses
- Detect threshold effects in environmental variables

---

## 6. Code description

The `code/` folder contains three main scripts:

- `data_preprocessing.py`: builds the final dataset from raw gridded inputs
- `RF_model.py`: trains the Random Forest model and evaluates performance
- `SHAP_analysis.py`: generates SHAP importance plots and response curves

---

## 7. Reproducibility

### Installation

Install required dependencies using:

```bash
pip install -r requirements.txt
The environment includes commonly used scientific Python libraries such as numpy, pandas, scikit-learn, shap, and matplotlib.

## 8. Data availability

All datasets used in this study are derived from publicly available sources, including:

SoilGrids (ISRIC)
Climate reanalysis datasets
Satellite-derived radiation and atmospheric CO₂ products
Global agricultural and irrigation datasets

Processed data used for modeling are included in this repository to ensure reproducibility.

## 9. License

This project is released under the MIT License.

Users are free to:

Use
Modify
Distribute
Cite this work

with appropriate attribution.

## 10. Contact

Corresponding author:
[Chang’an Zhou]
[zhouchangan@sdau.edu.cn]
## Code and Data Availability

The source code and processed dataset supporting this study are publicly available at:

GitHub:
https://github.com/sdauning/soil-first-soybean-framework

A permanent archived version of this repository is available at:

Zenodo:
https://doi.org/10.5281/zenodo.21156720
