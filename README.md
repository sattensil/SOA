# SOA Predictive Analytics Exam (PA) Solutions

This repository contains solutions for the Society of Actuaries (SOA) Predictive Analytics (PA) exams, organized in folders by exam date. Each folder contains both official solutions and my own implementations.

## Project Structure

The repository is organized with separate folders for each exam session, named according to the exam date (YYYY_MM format).

### PA 201812 (December 2018)

This exam focuses on predicting injury rates in mining operations and is currently highlighted in this repository.

#### My Solution
- `Exam_PA_2018_12_XGB_&_GLM.ipynb`: My Jupyter notebook solution using XGBoost and GLM approaches
- `edu_2018_12_minimal.py`: My Python implementation of the solution

#### Official Solutions
- `edu-2018-12-exam-pa-rmd solution.Rmd`: Official R Markdown solution file
- `edu-2018-12-pa-exam-solutions.pdf`: Official PDF document with solutions

#### Data
- `MSHA_Mine_Data_2013-2016.csv`: Dataset used for the analysis

## Analysis Overview

The analysis focuses on predicting injury rates in mining operations using various features such as:

- Mine characteristics (type, commodity, etc.)
- Employee hours in different activities (underground, surface, office, etc.)
- Seam height and other operational factors

### Modeling Approaches

#### Official Solution Models
1. **Decision Tree**: Provides an interpretable model showing the most important factors affecting injury rates
2. **Poisson GLM**: Offers statistical insights into how different factors influence injury rates

#### My Notebook Models
1. **XGBoost**: Advanced gradient boosting implementation with hyperparameter tuning for optimal predictive performance
2. **Tweedie GLM**: Generalized Linear Model using Tweedie distribution, which is more appropriate for the mixed discrete-continuous nature of the target variable
3. **Model Comparison**: Comprehensive evaluation using multiple metrics including log-likelihood, MSE, and cross-validation

## Key Findings

- Underground work is associated with significantly higher injury rates
- Office work is associated with lower injury rates
- Larger operations tend to have slightly lower injury rates per hour worked

## Setup and Installation

This project uses Poetry for dependency management. To install dependencies:

```bash
poetry install
```

To run the Python solution:

```bash
poetry run python edu_2018_12_minimal.py
```

## Dependencies

See `pyproject.toml` for the complete list of dependencies.