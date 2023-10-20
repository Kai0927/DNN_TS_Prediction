# Afetrnoon Thuderstrom forecast by Machine Learning

#### Motivation
The traditional weather forecast makes it hard to predict the afternoon thunderstorm.
However, the afternoon thunderstorm impacts our lives pretty much!
Therefore, I hope DNN has a greater prediction ability.

#### Tools and Data
* DNN
* 5 years (2016~2021) CWA weather OBS station data for training.
* 1 year (2022) CWA weather OBS station data for testing.
  
#### Tips for training
* Batch normalization: due to the high variance between each parameter.
* VIF (Variance Inflation Factor): due to the multicollinearity problem.

#### Result
> Without VIF calculation
* the POD is only 22 % in the testing dataset.

> With VIF calculation
* The POD increases to 42 % in the testing dataset.

#### Brief summary
In this project, the preprocessing of data is quite important!
It can increase by about 20 % performance after having VIF calculation.

<img src="https://github.com/Kai0927/DNN_TS_Prediction/blob/main/image/Performance.png" width="50%" height="50%">

<img src="https://github.com/Kai0927/DNN_TS_Prediction/blob/main/image/DNN_Prediction_vs_OBS.png" width="100%" height="100%">
