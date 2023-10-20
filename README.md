# Afetrnoon thuderstrom forecast by Machine Learning

#### Motivation
The traditional weather forecast is hard to predict the afternoon thunderstorm.
However, afternoon thunderstorm impacts our life pretty much!
Therefore, I hope that DNN can has the greater orediction ability.

#### Tools and Data
* DNN
* 5 years (2016~2021) CWA wheather OBS station data for training.
* 1 year (2022) CWA wheather OBS station data for testing.
  
#### Tips for training
* Batchnormalization: due to the high variance between each parameter.
* VIF (Variance Inflation Factor): due to the multicollinearity problem.

#### Result
> Without VIF calculation
* the POD is only 22 % in the testing dataset.

> With VIF calculation
* the POD increases to 42 % in the testing dataset.

#### Brief summary
In this project, the preprocessing of data is quite important!
It can increase about 20 % preformance after having VIF calculation.
