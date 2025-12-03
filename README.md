| ⚡ 2-layer NN | 🎯 MAE 2.18 y | 📊 RMSE 8.45 y |
|--------------|---------------|----------------|

**WHO Life-Expectancy Predictor**

A minimal 2-layer neural network that learns 15 years of global health data (2000-2015) to estimate how long people live.

**What it does**

Feed the model 21 numbers—immunisation coverage, GDP, schooling years, alcohol use, BMI, HIV prevalence, etc.—and it returns life expectancy in years.  
Architecture: 128 ReLU neurons → 1 linear output, built with Keras/TensorFlow 2.x.  
No fancy tuning: one learning-rate, 40 epochs, batch-size 1. Purposefully simple so anyone can fork and hack.  

**Results**  
Mean absolute error ≈ 2 years → on average the model misses by only 2 birthdays.  
RMSE > MAE tells us a handful of countries are harder to predict (outliers in HIV, war, famine).  
2-year error is within WHO’s own annual revision band, so the net is capturing the main signal.  

