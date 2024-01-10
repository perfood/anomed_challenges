# AnoMed Challenges
## Challenge 1
Straightforward time series prediction task.

Inputs: Macronutrients (carbohydrate, protein, fat, fiber), 2D circular time encoding, sex (0: female, 1: male), age, BMI, mean blood glucose

Target: Blood glucose reaction over 3 hours (15 min intervals)

Test Metrics:

    MSE: 308.62 mg²/dl²
    MAE: 12.37 mg/dl
## Challenge 2
Similar to challenge 1, but with a different target making it a regression task.

Inputs: Macronutrients (carbohydrate, protein, fat, fiber), 2D circular time encoding, sex (0: female, 1: male), age, BMI, mean blood glucose

Target: Blood glucose reaction area under the curve (AUC) and delta_max

Test Metrics:

    AUC:
        MSE: 610.05 mg²/dl²
        MAE: 17.55 mg/dl
    delta_max:
        MSE: 403.18 mg²/dl²
        MAE: 15.06 mg/dl
## Challenge 3
In this challenge, we are given days of continuous glucose measurements and meal data. The task is to transform these into features and predict the blood glucose reaction over a period of 3 hours after a meal, with data sampled at 15-minute intervals.

Inputs: Full glucose runs, macronutrients (carbohydrate, protein, fat, fiber), meal time

Target: Blood glucose reaction over 3 hours (15 min intervals)

Test Metrics:

    MSE: 272.83 mg²/dl²
    MAE: 11.66 mg/dl
## Challenge 4
Similar to challenge 3, but without preceding glucose measurements for the test meals.

Inputs: Full glucose runs (up to 1 day before the test meal), macronutrients (carbohydrate, protein, fat, fiber), meal time

Target: Blood glucose reaction over 3 hours (15 min intervals)

Test Metrics:

    MSE: 340.34 mg²/dl²
    MAE: 13.15 mg/dl
## Challenge 5
In this challenge, we predict the time to peak using the meal data, anamnesis data and the mean time to glucose peak. 

Inputs: Macronutrients (carbohydrate, protein, fat, fiber), 2D circular time encoding, sex (0: female, 1: male), age, BMI, height, weight, mean time to glucose peak (ttp_mean)

Target: Time to glucose peak

Test Metrics: 

    MSE: 1058.51 minutes
    MAE: 26.12 minutes
## Challenge 6
In this challenge we predict anamnesis features based on continuous glucose measurements, meal data and the delta_max value.

Inputs: Full glucose runs, meals with macronutrients (carbohydrate, protein, fat, fiber), meal time, delta_max

Targets: Sex (0: female, 1: male), weight, age.

Test RMSE: 

    Sex: 0.411
    Weight: 17.60 kg
    Age: 11.18 years
## Challenge 7
This challenge focuses on imputing missing anamnesis features using a combination of continuous glucose measurements, meal data, and other patient characteristics.

Inputs: Age, sex, weight, blood glucose quantile features (q0.4, q0.75, q0.9)

Targets: Each of the input features has missing elements that need to be imputed.

Test metrics (usually RMSE):

    Age: 10.26 years
    Sex: Accuracy of 0.720
    Weight: 17.37 kg
    q0.4: 4.16 mg/dl
    q0.75: 1.69 mg/dl
    q0.9: 3.92 mg/dl
