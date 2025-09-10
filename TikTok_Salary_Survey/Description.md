## Project Overview: Salary Survey Analysis and Modeling
![image](https://github.com/v-acha/Data_Science_Projects/assets/166547727/07f3d5ea-50cc-4c8f-a02b-235c0eb25578)

### Objective:
The primary goal of this project was to conduct a comprehensive analysis of salary data collected through surveys from TikTok users. The intent was to uncover insights into salary distributions, bonus structures, and the impact of various factors such as education, industry, job title, and geography on compensation. The next focus was developing models to predict salary and gender based on multiple variables.

### Data Aggregation:
Once data was collected, the files were processed through an SSIS package and SQL server. Then, using Pandas, the processed survey data from multiple files were compiled into a single dataset from various CSV files, creating a substantial data pool for analysis.

### Data Dictionary (dataset columns)
The following variables were collected: 
| Variable                           | Description                                                             |
|------------------------------------|-------------------------------------------------------------------------|
| Age Range                          | Categorical, representing the participant's age group.                  |
| Experience                         | Numerical, indicating how many years of professional experience the participant has.|
| Industry                           | Categorical, showing the industry in which the participant works.       |
| Job Title                          | Categorical, listing the participant's job title.                        |
| Education                          | Categorical, detailing the highest level of education the participant has completed.|
| Country                            | Categorical, specifying the country of the participant's employment.     |
| Annual Salary                      | Numerical, the participant's annual salary.                             |
| Annual Bonus                       | Numerical, the annual bonus amount the participant receives, if any.     |
| Signon Bonus                       | Numerical, the sign-on bonus amount the participant received upon hiring, if any.|                                                   |
| Gender                             | Categorical, indicating the participant's gender.|

### Data Cleaning and Preprocessing: 
The project's data cleaning phase was comprehensive, involving several steps to ensure uniformity and accuracy:
- **Standardizing Columns:** Renaming and correcting columns for categorical consistency.
- **Categorization and Grouping:** Categories were created for 'Education', 'Industry', 'Job Title', and 'Gender' based on the survey responses. Education levels were grouped into broad categories, and similar approaches were taken for industries and job titles.
- **Web Scraping and Fuzzy Matching:** Lists of standardized industry and job titles were scraped from the Bureau of Labor Statistics website. Fuzzy matching was then employed to align survey data with these lists.
- **Country Data Standardization:** Country names were corrected and standardized.
- **Salary Conversion:** Salaries were converted into USD using country-specific conversion rates to ensure consistent financial analysis.

### Findings
The analysis uncovers key salary trends influenced by location, age, experience, industry, and job title. It highlights geographical salary variations and a positive correlation between age/experience and earnings. Some industries prioritize practical skills over formal education, impacting compensation. Certain job titles command higher pay due to specialized roles. Overall, the findings emphasize the complex interplay of these factors in determining salaries.

### XGBoost Model for Salary prediction:
Chosen for its capability to capture complex relationships within the data.
Hold-out Test Set Performance
- **Hold-out Test Set MSE:** The MSE on the hold-out test set is approximately 446 million, which is slightly better (lower) than the mean CV MSE. This suggests that the model generalizes well to unseen data, at least regarding the MSE metric.
- **Hold-out Test Set R-Square (R²):** The R² value of 0.602 indicates that the model explains about 60.22% of the salary variance. This is a decent level of predictive power, suggesting the model has learned meaningful patterns from the features that contribute to salary prediction.
- **Model Performance:** The gradient boosting model shows a good balance between bias and variance, as indicated by the consistency of CV MSE scores and the reasonable R² value on the hold-out test set. An R² of over 0.60 indicates a model that captures a significant portion of the variance in the target variable.

### Random Classification Model for Gender:
The model achieves an overall accuracy of 81%, indicating a reasonable capability to predict gender based on the provided features. However, the performance varies across different gender classes:
- **Female:** High precision (83%) and recall (97%), suggesting excellent model performance for this class.
- **Male:** Moderate precision (61%) but low recall (19%), indicating the model struggles to identify this class correctly.
- **Non-Binary:** Moderate precision (33%) but low recall (6%), showing the model's difficulty in identifying this class.
![image](https://github.com/v-acha/Data_Science_Projects/assets/166547727/375f3158-2b03-48ab-b910-7256ab8e42f8)
While the model performs well for the Female class, it exhibits challenges in accurately predicting the Male and Non-Binary classes. This is probably caused by the disproportionate distribution of gender in the dataset. With a significantly larger sample size for females (37,088) compared to males (9,320) and non-binary (81) individuals, the model is biased towards predicting the majority class (females) more accurately while struggling with minority classes (males and non-binary individuals) due to the limited amount of data available for training on these groups. One significant challenge was managing the imbalanced dataset for gender prediction.

### Conclusions: 
During the analysis, certain correlations were discovered between salary, experience, education, and industry. This emphasized the value of exploratory data analysis in revealing important insights. The use of machine learning models for salary prediction and gender classification further highlighted the significance of tailoring model selection to the data's characteristics and objectives. However, there were certain challenges such as managing class imbalances and balancing model complexity with interpretability.

While the project achieved its primary data cleaning and analysis goal, predictive modeling accuracy can be enhanced. By refining methodologies and exploring advanced techniques, such as NLP and machine learning, addressing class imbalance, and refining standardization of categorical variables, future iterations of this project can offer even greater insights into salary dynamics and higher accuracy and precision in predictive modeling.
