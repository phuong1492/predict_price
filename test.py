import pandas as pd
from prophet import Prophet

# Read the CSV file
df = pd.read_csv('bnb_5m_data.csv')

# Convert start_time to datetime format
df['ds'] = pd.to_datetime(df['start_time'], unit='ms')

# Rename 'close' to 'y' for Prophet (the target variable)
df = df[['ds', 'close', 'volume', 'number_of_trades']].rename(columns={'close': 'y'})

# Initialize Prophet model
model = Prophet()

# Add additional regressors (e.g., volume, number_of_trades)
model.add_regressor('volume')
model.add_regressor('number_of_trades')

# Fit the model with historical data and additional regressors
model.fit(df)

# Predict the next 20 minutes
future = model.make_future_dataframe(periods=1, freq='20T')

# You need to supply reasonable future values for the regressors
# This example assumes future volume and number_of_trades are similar to recent data

# Get the last known volume and number_of_trades
last_known_volume = df['volume'].iloc[-1]
last_known_trades = df['number_of_trades'].iloc[-1]

# Use these as placeholders for future regressors
future['volume'] = last_known_volume
future['number_of_trades'] = last_known_trades

# Make predictions
forecast = model.predict(future)

# Output only the predicted value (yhat)
predicted_price = forecast['yhat'].iloc[-1]

# Print the predicted price
print(predicted_price)