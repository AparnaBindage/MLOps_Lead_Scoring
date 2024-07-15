# Project Name - Lending Club Case Study
Lending Club company is the largest online loan marketplace, facilitating personal loans, business loans, and financing of medical procedures. In this project we need to identify risky loan applicants, thereby to cut down the amount of credit loss. Identification of such applicants using EDA is the aim of this case study.
The goal is to build a machine learning model that is able to predict churning customers based on the features provided for their usage.

## Table of Contents
* [General Information](#general-information)
* [Steps Performed](#steps-performed)
* [Business Insights Derived](#Business-insights-derived)
* [Recommendations](#recommendations)
* [Libraries Used](#libraries-used)


## General Information
In the telecommunication industry, customers tend to change operators if not provided with attractive schemes and offers. It is very important for any telecom operator to prevent the present customers from churning to other operators. As a data scientist, your task in this case study would be to build an ML model which can predict if the customer will churn or not in a particular month based on the past data.

In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business
goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, you will analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn.


## Steps Performed
1. Data understanding
2. Data cleaning 
3. Outlier treatment
4. Data Analysis - Univariate and Bivariate
5. Model Building

## Business Insights Derived
1. Customers who recharge with amount 50 or less than 50 tend to churn more.
2. There is more chance that the customer is likely to churn if the total incoming minutes of usage is lesser in the month of August than any other month
3. Roaming outgoing minutes of usage is more for churn customers. Higher the minutes of usage, churn probability is higher
4. Customers having decreasing local outgoing minutes of usage for operators T to other operator mobile M for August are more likely to churn.

## Recommendations
1. Customers who recharge with amount 50 or less than 50 tend to churn more. It is suggested to give some attractive offers on the recharge plan.
2. Target the customers, whose minutes of usage of the total incoming calls are less in the action phase i.e. in August
3. It is suggested that more attractive plans should be given to customers in roaming zone
4. Target the customers whose local outgoing minutes of usage for operators T to other operator mobile M are less

## Libraries Used
- pandas
- plotly 
- matplotlib 
- seaborn

## Contact
Created by AparnaBindage - feel free to contact me!