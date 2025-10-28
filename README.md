# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# Read the Gold dataset from a CSV file
data = pd.read_csv('Gold Price Prediction.csv')  # Replace with the correct file path

# Display the shape and the first 20 rows of the dataset
print("Shape of the dataset:", data.shape)
print("First 20 rows of the dataset:")
print(data.head(20))

# Set the figure size for plots
plt.rcParams['figure.figsize'] = [10, 6]

# Plot the first 50 values of the 'Price Today' column
plt.plot(data['Price Today'][:50])
plt.title('First 50 Values of "Price Today"')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.grid(True)
plt.show()

# Perform rolling average transformation with a window size of 5 on 'Price Today'
rolling_mean_5 = data['Price Today'].rolling(window=5).mean()

# Display the first 10 values of the rolling mean
print("First 10 values of the rolling mean (window size 5):")
print(rolling_mean_5.head(10))

# Perform rolling average transformation with a window size of 10 on 'Price Today'
rolling_mean_10 = data['Price Today'].rolling(window=10).mean()

# Plot the original data and fitted value (rolling mean with window size 10)
plt.figure()
plt.plot(data['Price Today'], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)', color='orange')
plt.title('Original Data and Rolling Mean (window=10)')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.legend()
plt.grid(True)
plt.show()

# Perform exponential smoothing on 'Price Today' and plot the graph
alpha = 0.3  # Smoothing factor
exp_smooth = data['Price Today'].ewm(alpha=alpha).mean()

plt.figure()
plt.plot(data['Price Today'], label='Original Data')
plt.plot(exp_smooth, label='Exponential Smoothing', color='red')
plt.title('Original Data and Exponential Smoothing')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.legend()
plt.grid(True)
plt.show()

# Implement the Moving Average (MA) Model
# Set the order (p, d, q) to (0, 0, q) for a pure MA model. Let's try q = 1 first.
q = 1
ma_model = ARIMA(data['Price Today'], order=(0, 0, q))
ma_model_fit = ma_model.fit()

# Display the summary of the fitted model
print(ma_model_fit.summary())

# Forecast the next 10 data points using the fitted model
forecast = ma_model_fit.forecast(steps=10)

# Plot the original data and the forecasted values
plt.figure()
plt.plot(data['Price Today'], label='Original Data')
plt.plot(range(len(data), len(data) + 10), forecast, label='MA Forecast', color='green', marker='o')
plt.title('Original Data and Moving Average Forecast')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.legend()
plt.grid(True)
plt.show()

# Print the forecasted values
print("Forecasted values for the next 10 points:")
print(forecast)
```
### OUTPUT:
<img width="658" height="366" alt="image" src="https://github.com/user-attachments/assets/de924ae9-8fe8-4df3-87a7-7703ea90662e" />
<img width="1071" height="419" alt="image" src="https://github.com/user-attachments/assets/8b64aa1b-688b-4d90-b31d-f16797feb536" />
<img width="1069" height="700" alt="image" src="https://github.com/user-attachments/assets/8d3c4a9f-99c4-4d75-a353-8cbb1ce601c2" />
<img width="1052" height="726" alt="image" src="https://github.com/user-attachments/assets/8a634abe-66b5-4e20-850c-b9051675ebe5" />
<img width="1053" height="711" alt="image" src="https://github.com/user-attachments/assets/cdbc678f-712c-43f4-b4b3-2468c2e5acc2" />

Moving Average
<img width="1073" height="623" alt="image" src="https://github.com/user-attachments/assets/d7335870-8edf-43ea-a985-060e3869f99f" />

<img width="1035" height="626" alt="image" src="https://github.com/user-attachments/assets/718f282d-2f36-43c6-814a-e9c8403fe810" />




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
