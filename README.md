# AnoMed Challenges
## Challenge 1
Straightforward time series prediction task.

Inputs: Macronutrients (carbohydrate, protein, fat, fiber), 2D circular time encoding, sex, age, BMI, mean blood glucose

Target: Blood glucose reaction over 3 hours (15 min intervals)

Test Metrics: MSE: 305.37 mg²/dl², MAE: 12.33 mg/dl
## Challenge 2
Similar to challenge 1, but with a different target making it a regression task.

Inputs: Macronutrients (carbohydrate, protein, fat, fiber), 2D circular time encoding, sex, age, BMI, mean blood glucose

Target: Blood glucose reaction area under the curve (AUC) and delta_max

Test Metrics AUC: MSE: 558.88 mg²/dl², MAE: 16.98 mg/dl
Test Metrics delta_max: MSE: 379.34 mg²/dl², MAE: 14.54 mg/dl
## Challenge 3
In this challenge, we are given days of continuous glucose measurements and meal data. The task is to transform these into features and predict the blood glucose reaction over a period of 3 hours after a meal, with data sampled at 15-minute intervals.

Inputs: Full glucose runs, macronutrients (carbohydrate, protein, fat, fiber), meal time

Target: Blood glucose reaction over 3 hours (15 min intervals)

Test Metrics: MSE: 272.83 mg²/dl², MAE: 11.66 mg/dl
