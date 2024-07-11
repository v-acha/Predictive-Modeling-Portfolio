# Project Overview: Predictive Pricing Model
![image](https://github.com/v-acha/Data_Science_Projects/assets/166547727/c0cb5c5d-7021-4cb3-86d4-bf2f60789bea)

## Introduction
This document provides an overview of the data cleaning, exploratory data analysis (EDA), feature engineering, and modeling processes used to develop a predictive pricing model. The aim is to estimate the prices of products based on various features. 

## Objective
The primary objective of this project is to develop a predictive pricing model that accurately estimates the prices of shoes based on various product features. This model aims to provide actionable insights for pricing strategies, helping businesses optimize their pricing to maximize revenue and market competitiveness. Through thorough data cleaning, feature engineering, and the application of advanced modeling techniques, the project seeks to identify the key factors that influence shoe prices and utilize them to predict prices effectively.

The following steps were undertaken from data cleaning to model evaluation.
## Data
- Women’s shoe prices: 7003_1.csv (https://data.world/datafiniti/womens-shoe-prices)
- Men’s shoe prices: 7004_1.csv (https://data.world/datafiniti/mens-shoe-prices)

The datasets are a list of over 40,000 women's and men’s shoes and their product information provided by Datafiniti's Product Database. They include shoe name, brand, price, and more. Each shoe will have an entry for each price found for it and some shoes may have multiple entries.

## Data Cleaning and Feature Engineering

### Initial Steps

- **`brand`:** Converted all entries to uppercase and trimmed spacing.

- **`categories`:** Created a new column `shoe_category` to categorize "Women's Shoes" and "Men's Shoes".

- **date columns:** Converted `dateAdded`, `prices.dateSeen`, and `dateUpdated` to date values.

- **`colors`:** Filled missing values with "Unknown" or "No Color".

- **`prices.condition`:** Filled missing values with "new" as it is the most common value.

- **`name`:** Retained this high cardinality column for adding detail and precision to the model.

- **Prices columns:** 
    - Converted `prices.amountMin` and `prices.amountMax` to numeric values.
    - Removed outliers based on summary statistics.
    - Calculated the log-transformed scale for both min and max prices.

- **`average price`:** Created an `avg_price` column as the average of `prices.amountMin` and `prices.amountMax` to serve as the target variable for predictive model.

- **`currency`:** Removed nulls and categorized if needed.

- **`sale status`:** Standardized `prices.isSale` into TRUE or FALSE categories.

- **`manufacturer number`:** Filled NaN values with "Unknown".

- **`merchant_source`:** Created a new column `merchant_source` with cleaned merchant names.
    - Dropped the `prices.sourceURLs` column after cleaning.
    - `clean_url` function: Removed specific escape sequences from URLs, ensuring standardized formats.
    - `extract_merchant_name` function: Extracted base domain names to represent the merchant, logging unprocessable URLs into separate CSV files for women’s and men’s datasets.

**data verification and final cleaning:**
- Verified and standardized all columns.
- Explore price statistics and dropped outliers and items not related to shoes.
- Combined initial dataset of 38,432 rows (19,045 for women and 19,387 for men) and 47 columns each.
- Final cleaned dataset: 36,213 rows combined, 17 columns.

## Exploratory Data Analysis
- Conducted EDA using pair plots and correlation heatmap to understand relationships between variables.
- Principal Component Analysis (PCA) was applied during the exploratory data analysis (EDA) phase to understand the underlying structure and relationships within the dataset. 

During the exploratory data analysis, we found strong positive correlations between the minimum, maximum, and average prices of shoes. The correlation matrix confirmed these relationships, with values close to 1. PCA visualization revealed that most shoes are clustered within a similar price range, with distinct clusters for higher-priced shoes. This indicates that the dataset predominantly consists of moderately priced shoes with a few high-priced outliers.

## Modeling
Developed and evaluated the predictive pricing model using the cleaned dataset. 

Random Forest Regression and Ridge Regression were selected based on findings from Exploratory data anlysis. As they are well suited to effectively handle correlated features, multicollinearity, and outliers while enhancing model robustness and interpretability.

### Model 1. Random Forest Regressor:
The model utilized a RandomForestRegressor with 50 estimators. This model was trained using a combination of categorical and numerical features to predict product prices. The following features were used: `brand`, `categories`, `colors`, `prices.amountMin`, `prices.amountMax`, `prices.condition`, `shoe_category`, `name`, `merchant_source`, `prices.isSale`. 

**Model Performance:** The model achieved a Mean Absolute Error (MAE) of 0.07153.

**Interpretation:** This means, on average, the model's predictions deviate from the actual prices by 0.0730 units of the currency. Predicted prices are off by only 0.0730 units (e.g., dollars, euros) from the actual prices.

### Model 2. Linear Regression: 
The Linear Regression model achieved a Mean Absolute Error (MAE) of 7.1357.

### Model 3. Ridge Regression

**Model Performance:** The model achieved a Mean Absolute Error (MAE) of 28.9610 using `avg_price` alone, 0.0159 using `prices.amountMin` and `prices.amountMax`, and 0.0106 using all three features.

**Interpretation:** This means that relying solely on `avg_price` for predictions results in a significantly higher error, indicating less accuracy. Utilizing `prices.amountMin` and `prices.amountMax` drastically reduces the error, providing a much more precise prediction model. Incorporating `avg_price` along with `prices.amountMin` and `prices.amountMax` further enhances the model's accuracy, leading to the most reliable price predictions. Therefore, the best approach is to use all three features for the most accurate pricing predictions.

## Conclusion
After conducting extensive data cleaning, feature engineering, and evaluating multiple predictive models, we assessed the performance of Random Forest Regression, Linear Regression, and Ridge Regression for price modeling. Here are the key findings:

- **Random Forest Regression**: This model provided robust predictions with reasonable accuracy but was computationally intensive and slower to run. While it handles non-linear relationships and interactions between features well, its complexity and longer training times make it less practical for large datasets or real-time applications.

- **Linear Regression**: Although simple and easy to interpret, the Linear Regression model struggled with multicollinearity and provided less accurate predictions compared to the other models. Its performance was not as competitive, making it a less favorable choice for this specific price modeling task.

- **Ridge Regression**: The Ridge Regression model emerged as the best choice for price modeling. It achieved the lowest Mean Absolute Error (MAE) of 0.0106 when using all three features (`avg_price`, `prices.amountMin`, and `prices.amountMax`). This model not only provided the most accurate predictions but also effectively handled multicollinearity, which is a common issue in datasets with correlated features. Additionally, Ridge Regression runs faster than both Random Forest and Linear Regression, making it highly efficient and suitable for real-time pricing applications.

**Best Choice for Price Modeling**: Based on the evaluation results, Ridge Regression is the optimal choice for price modeling. It combines accuracy, efficiency, and the ability to manage multicollinearity, ensuring reliable and precise price predictions while maintaining computational efficiency. This makes Ridge Regression the preferred model for both current and future price prediction tasks.
