import pandas as pd
import pandas_ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# loading data into a variable df using read_csv
df = pd.read_csv('TSLA.csv')

# print(df)

# Reindex data using a DatetimeIndex
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

# Keep only the 'Adj Close' Value
df = df[['Adj Close']]

# Re-inspect data
# print(df)

# Print Info
print(df.info())

# Add EMA to dataframe by appending
# Note: pandas_ta integrates seamlessly into
# our existing dataframe
df.ta.ema(close='adj_close', length=10, append=True)

# Drop the first n-rows
df = df.iloc[10:]


# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(df[['adj_close']], df[['EMA_10']], test_size=.2)

# Test set
print(X_test.describe())

# Training set
print(X_train.describe())

# Create Regression Model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)


print("Model Coefficients:", model.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))

