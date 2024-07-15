# Project Name - Lead Scoring Case Study
Code pro is an edtech startup that had a phenomenal seed A funding round. They used this funding to gain significant market share by campaigning aggressively. Due to aggressive campaigns the amount of junk leads increased.

A lead is generated when any person visits Code pro’s website and enters their contact details on the platform. A junk lead is when the person sharing his contact details has no interest in the product/service.

Having junk leads in the pipeline creates a lot of inefficiency in the sales process. Thus the goal of the Data Science team is to build a system that categorizes leads on their propensity to purchase Code pros course. This system will help remove the inefficiency caused by junk leads in the sales process.

## Table of Contents
* [General Information](#general-information)
* [Pipelines Implemented](#pipelines-implemented)
* [Libraries Used](#libraries-used)
* [Acknowledgements](#acknowledgements)

## General Information
CodePro is an EdTech startup that used the money to increase its brand awareness. As the marketing spend increased, it got several leads from different sources. Although it had spent significant money on acquiring customers, it had to be profitable in the long run to sustain the business. The major cost that the company is incurring is the customer acquisition cost (CAC).

At the initial stage, customer acquisition cost is required to be high in companies. But as their businesses grow, these companies start focussing on profitability. Many companies first offer their services for free or provide offers at the initial stages but later start charging customers for these services.

Businesses want to reduce their customer acquisition costs in the long run. The high customer acquisition cost is due to following reasons:

1. Improper targeting
2. High competition
3. Inefficient conversion

To address inefficient conversion, the sales team must undergo upskilling and prioritise the leads generated.
The sales team must work with the data science team to figure out how to prioritise leads. The data science team must come up with lead scoring for all the leads generated.

## Pipelines Implemented
1. Development Pipeline
- Rapid Experimentation is performed using pycaret and tracked using MLflow to build baseline model
2. Production Pipeline
- Data Pipeline - To fetch data from source and preprocess it for training and inference pipeline
- Training Pipeline - In case of data drift under a threshold, retrain the model with new preprocessed data
- Inference Pipeline - To predict the target variable for new data

## Libraries Used
- MLflow 
- Airflow
- PyCaret
- Pandas Profiling
- Python3
- numpy
- pandas
- matplotlib 
- Jupyter

## Acknowledgements
Thankful to the professors of Upgrad & IIIT, Bangalore for providing the necessary knowledge & support to accomplish this project. 
