# AnoMed Challenges
## Challenge 1: Time Series Prediction of Blood Glucose Reaction

### Task Description
The objective of Challenge 1 is to perform a time series prediction of blood glucose reaction over a 3-hour period following a meal, with data sampled at 15-minute intervals. This is a regression task where the model is expected to predict continuous values representing the blood glucose levels.

### Input Features
The dataset for this challenge includes the following features:
- Macronutrients: Carbohydrate, Protein, Fat, Fiber (numeric values)
- 2D Circular Time Encoding: Represents the time of the day (numeric, 2-dimensional)
- Sex: Binary value (0 for female, 1 for male)
- Age: Numeric value representing the age of the individual
- BMI: Body Mass Index (numeric value)
- Mean Blood Glucose: Average blood glucose level (numeric value)

### Output Target
The target variable is a vector representing the blood glucose reaction at 15-minute intervals over a span of 3 hours post-meal. This results in a total of 12 predictions per instance, corresponding to the 12 time intervals.

### Model Input and Output Dimensions
- Input Dimension: The model should accept an input vector of size 10.
- Output Dimension: The model should output a vector of size 12, each element representing the predicted blood glucose level at a specific time interval.

### Test Metrics
The model as used in the reference code produces the following test metric results:
- Mean Squared Error (MSE): 308.62 mg²/dl²
- Mean Absolute Error (MAE): 12.37 mg/dl
## Challenge 2: Regression Analysis of Blood Glucose Reaction Metrics

### Task Description
The goal of Challenge 2 is to conduct a regression analysis to predict two distinct blood glucose reaction metrics: the area under the curve (AUC) and the maximum change in blood glucose level (delta_max). These metrics are crucial for understanding the overall blood glucose reaction post-meal.

### Input Features
The dataset for this challenge includes the following features:
- Macronutrients: Carbohydrate, Protein, Fat, Fiber (numeric values)
- 2D Circular Time Encoding: Represents the time of the day (numeric, 2-dimensional)
- Sex: Binary value (0 for female, 1 for male)
- Age: Numeric value representing the age of the individual
- BMI: Body Mass Index (numeric value)
- Mean Blood Glucose: Average blood glucose level (numeric value)

### Output Target
Two models can be used to predict the two separate target variables:
1. The area under the curve (AUC) for blood glucose reaction, which integrates the glucose level over time.
2. The maximum change in blood glucose level (delta_max) from baseline.

### Model Input and Output Dimensions
- Input Dimension: The models should accept an input vector of size 10.
- Output Dimension: Each model should output one values (AUC and delta_max).

### Test Metrics
The models as used in the reference code produce the following metrics on the test data:
- For AUC:
  - Mean Squared Error (MSE): 610.05 mg²/dl²
  - Mean Absolute Error (MAE): 17.55 mg/dl
- For delta_max:
  - Mean Squared Error (MSE): 403.18 mg²/dl²
  - Mean Absolute Error (MAE): 15.06 mg/dl
## Challenge 3: Blood Glucose Reaction Prediction with Historical Glucose Measurements

### Task Description
The goal of Challenge 3 is to predict the blood glucose reaction within a 3-hour window following meal consumption, with data points collected at 15-minute intervals. This task involves a regression problem where the model must forecast continuous blood glucose levels for the specified duration.

### Input Features
The dataset provided for this challenge comprises the following input features:
- Continuous Glucose Measurements: Historical glucose levels (numeric values)
- Macronutrients: Quantities of Carbohydrate, Protein, Fat, and Fiber from meal data (numeric values)
- Meal Timing: Specific time of day when the meal was consumed (numeric value)

### Output Target
The output target is a series of blood glucose level predictions at 15-minute intervals for 3 hours post-meal. This equates to 12 predicted values for each instance, each corresponding to one of the 12 time intervals.

### Model Input and Output Dimensions
- Input Dimension: The model receives an input vector with a size that corresponds to the number of features derived from the glucose measurements, macronutrient content, and meal timing.
- Output Dimension: The model outputs a vector with 12 elements, each representing the predicted blood glucose level at a subsequent 15-minute interval.

### Test Metrics
The reference implementation of the model for this challenge achieved the following test metrics:
- Mean Squared Error (MSE): 272.83 mg²/dl²
- Mean Absolute Error (MAE): 11.66 mg/dl
## Challenge 4: Blood Glucose Reaction Prediction Without Prior Glucose Measurements

### Task Description
Challenge 4 extends the predictive modeling of Challenge 3 by forecasting blood glucose reactions within a 3-hour window post-meal without the benefit of preceding glucose measurements for the test meals. This scenario presents a more challenging regression task where the model must infer blood glucose levels based solely on full glucose runs up to a day before the test meal, macronutrient intake, and meal timing.

### Input Features
The dataset provided for this challenge comprises the following input features:
- Continuous Glucose Measurements: Historical glucose levels up to 24 hours before the test meal (numeric values)
- Macronutrients: Quantities of Carbohydrate, Protein, Fat, and Fiber from meal data (numeric values)
- Meal Timing: The exact time of day the meal was consumed (numeric value)

### Output Target
The output target is a series of blood glucose level predictions at 15-minute intervals for 3 hours post-meal. This equates to 12 predicted values for each instance, each corresponding to one of the 12 time intervals.

### Model Input and Output Dimensions
- Input Dimension: The model receives an input vector with a size that corresponds to the number of features derived from the glucose measurements, macronutrient content, and meal timing.
- Output Dimension: The model outputs a vector with 12 elements, each representing the predicted blood glucose level at a subsequent 15-minute interval.

### Test Metrics
The reference implementation for Challenge 4 achieved the following test metrics:
- Mean Squared Error (MSE): 340.34 mg²/dl²
- Mean Absolute Error (MAE): 13.15 mg/dl
## Challenge 5: Prediction of Time to Glucose Peak

### Task Description
The objective of Challenge 5 is to predict the time to peak glucose levels following a meal. This prediction is based on meal composition, patient anamnesis data, and the mean time to glucose peak. Accurate prediction of the time to peak glucose is important for managing postprandial blood glucose levels in individuals.

### Input Features
The dataset for this challenge comprises the following input features:
- Macronutrients: Carbohydrate, Protein, Fat, Fiber (numeric values)
- 2D Circular Time Encoding: Encodes the time of day (numeric, 2-dimensional)
- Sex: Binary value (0 for female, 1 for male)
- Age: Numeric value representing the individual's age
- BMI: Body Mass Index (numeric value)
- Height: Numeric value representing the individual's height
- Weight: Numeric value representing the individual's weight
- Mean Time to Glucose Peak (ttp_mean): Average time to reach peak glucose level (numeric value)

### Output Target
The model aims to predict a single target variable:
- Time to Glucose Peak: The duration it takes to reach the maximum glucose level after a meal.

### Model Input and Output Dimensions
- Input Dimension: The models should accept an input vector of size 12.
- Output Dimension: The model should output a single value representing the time to glucose peak.

### Test Metrics
The model's performance on the test data is evaluated using the following metrics:
- Mean Squared Error (MSE): 1058.51 minutes²
- Mean Absolute Error (MAE): 26.12 minutes
## Challenge 6: Predicting Anamnesis Features from Glucose and Meal Data

### Task Description
The objective of Challenge 6 is to predict some anamnesis features using continuous glucose monitoring data, meal composition, and the maximum change in blood glucose level (delta_max). This challenge focuses on understanding the relationship between an individual's dietary intake, glucose response, and personal health attributes.

### Input Features
The dataset for this challenge comprises the following input features:
- Full Runs of Continuous Glucose Measurements
- Meal Data: Macronutrient content including carbohydrate, protein, fat, and fiber, time of meal
- Blood Glucose Reaction Metric: Maximum change in blood glucose level (delta_max) for each meal

### Output Target
The models are designed to predict the following anamnesis features:
1. Sex: Categorized as 0 for female and 1 for male
2. Weight: Numeric value representing the individual's weight in kilograms
3. Age: Numeric value representing the individual's age in years

### Model Input and Output Dimensions
- Input Dimension: These reference models accept an input vector that includes glucose quantiles, macronutrient averages, and correlation of these averages with delta_max.
- Output Dimension: Each model outputs a single value corresponding to one of the target anamnesis features (Sex, Weight, or Age).

### Test Metrics
The performance of the models on the test data is evaluated using the Root Mean Squared Error (RMSE) for each anamnesis feature:
- For Sex:
  - RMSE: 0.411
- For Weight:
  - RMSE: 17.60 kg
- For Age:
  - RMSE: 11.18 years
## Challenge 7: Imputation of Missing Anamnesis Features

### Task Description
The objective of Challenge 7 is to accurately impute missing data within anamnesis features. This task is could be used for maintaining the integrity of patient records and ensuring that subsequent analyses are based on complete datasets.

### Input Features
The dataset for this challenge comprises the following features:
- Age: Numeric value representing the patient's age
- Sex: Binary value (0 for female, 1 for male)
- Weight: Numeric value representing the patient's weight in kilograms
- Blood Glucose Quantile Features: Quantitative representations of blood glucose levels at different quantiles (q0.4, q0.75, q0.9)

### Output Target
The target for this challenge is to impute the missing value for each of the input features listed above. The imputation model must predict the missing data points to complete the dataset.

### Model Input and Output Dimensions
- Input Dimension: The imputation models should accept an input vector that excludes the feature being imputed, leaving 5 input dimensions.
- Output Dimension: Each imputation model outputs a single value corresponding to the missing data point for the feature being imputed.

### Test Metrics
The imputation models used in the reference code yield the following metrics on the test data:
- For Age:
  - Root Mean Squared Error (RMSE): 10.26 years
- For Sex:
  - Accuracy: 72.0%
- For Weight:
  - Root Mean Squared Error (RMSE): 17.37 kg
- For Blood Glucose Quantile q0.4:
  - Root Mean Squared Error (RMSE): 4.16 mg/dl
- For Blood Glucose Quantile q0.75:
  - Root Mean Squared Error (RMSE): 1.69 mg/dl
- For Blood Glucose Quantile q0.9:
  - Root Mean Squared Error (RMSE): 3.92 mg/dl