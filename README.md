![image](https://github.com/v-acha/Data_Science_Projects/assets/166547727/45241d56-80c7-4206-857c-d7468ebfdca0)

# Predictive Modeling Projects
Welcome to the Data Science Project repository! This repository contains some of my datascience projects. They each offer insights and analyses across diverse domains. My projects range from exploratory data analysis to predictive modeling and machine learning applications, all aimed at solving problems, and driving informed decision-making and furthering my skill set. Each dataset is chosen based on my interests and the skills I want to learn.

1. [Data Analysis, Salary and Gender Prediction](https://github.com/v-acha/Data_Science_Projects/tree/main/TikTok_Salary_Survey)
   - **Description:** - This project entails analyzing salary data gathered from a TikTok survey to predict salary amounts based on various categories and gender, using specific variables as predictors.
   - **Technologies Used:** - The notebooks uses Pandas, NumPy, Matplotlib and Seaborn for data analysis and visualizations. XGBoost for salary prediction and Random Classification for gender prediction.
   - **Results:** - The XGBoost regression model achieved an R-squared of 0.60 for predicting salary amounts and The Random Classification model achieved an accuracy of 0.81 for predicting gender.

2. [Shoe Price Prediction](https://github.com/v-acha/Data_Science_Projects/tree/main/shoe_price_prediction)
   - **Description:** - This project involves analyzing shoe price data from datafiniti to predict average prices based on various categories and conditions using specific variables as predictors.
   - **Technologies Used:** - The notebooks use Pandas and NumPy for data manipulation, Matplotlib and Seaborn for data visualization, and Scikit-learn for applying PCA and building Random Forest regression, Linear regression, and Ridge regression models.
   - **Results:** - The Ridge regression model, incorporating all three features (prices.amountMin, prices.amountMax, and avg_price), achieved the lowest MAE of 0.0106, effectively predicting shoe prices with high accuracy. The Linear regression model had a higher MAE of 7.1357, while the Random Forest regression model achieved an MAE of 0.07153.
3. [Wildfire Ignition Risk Prediction (FireGuard)](https://github.com/v-acha/Data_Science_Projects/tree/main/shoe_price_prediction)
   - **Description** - This project forecasts where wildfires are most likely to ignite across California using 12 years of historical fire detections and daily environmental data. The goal is to produce forward-looking risk maps that support proactive wildfire prevention.
   - **Technologies Used** - The notebook uses AWS, Sagemaker notebook, sagemaker endpoins, cloud watch, s3bucket, Pandas, NumPy, Matplotlib, Seaborn for EDA and Scikit-learn, XGBoost, Tabnet, SHAP for modeling.
   - **Results** - The final model achieved strong performance with an ROC AUC of **0.91** and identified key risk drivers like temperature, humidity, and vegetation dryness. FireGuard outputs **daily spatial ignition risk maps** to guide early intervention and reduce fire spread.
4. [Corporate Bankruptcy Prediction](https://github.com/v-acha/Data_Science_Projects/tree/main/shoe_price_prediction)
   - **Description** - This project focuses on predicting whether a company will go bankrupt based on financial indicators. The model aims to improve early identification of at-risk firms, addressing a highly imbalanced classification task.
   - **Technologies Used** - 
      - **Modeling & Preprocessing:** Scikit-learn, TensorFlow (Keras), XGBoost, LightGBM  
      - **Class Imbalance Handling:** SMOTE, ADASYN, Balanced Random Forest (imbalanced-learn)  
      - **Feature Engineering:** Correlation filtering, standardization  
   - **Results** - After extensive evaluation, the **Balanced Random Forest** model with tuned hyperparameters achieved a **recall of 0.86** and a **ROC AUC of 0.94** on the test set. It successfully identified most bankruptcies while maintaining a **precision of 0.27**, offering practical value for financial risk screening and intervention.

