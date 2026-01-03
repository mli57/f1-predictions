import os
import fastf1
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

## Create local cache if not already existing
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)



## retrieve data
def get_quali_times(year):
    session = fastf1.get_session(year, 'Singapore', 'Q')
    session.load()

    # keep only meaningful laps (no outlaps)
    laps = session.laps.pick_quicklaps()

    # find fastest lap for each driver & remove NaT(not a time) from red flags, mechanical issues, etc.
    fastest = laps.groupby('Driver')['LapTime'].min().reset_index()
    fastest = fastest.dropna(subset=['LapTime'])

    # convert to seconds, XGBoost cannot handle Timedelta(duration, not number)
    fastest['LapTime_sec'] = fastest['LapTime'].dt.total_seconds()
    fastest['Year'] = year # assigns correct year to lap
    
    return fastest[['Driver', 'LapTime_sec', 'Year']]

def get_race_results(year):
    session = fastf1.get_session(year, 'Singapore', 'R')
    session.load()

    df = pd.DataFrame(session.results).copy()
    
    # Rename columns to more intuitive names
    rename_map = {'Abbreviation': 'Driver', 'TeamName': 'Constructor', 'GridPosition': 'Grid'}
    df.rename(columns=rename_map, inplace=True)
   
    df['Year'] = year # add reliable year column
    df = df[['Position', 'Driver', 'Constructor', 'Grid', 'Year']] # keep only these data
    
    return df

## Pull 2022-2024 data from api
years = [2022, 2023, 2024]

# build list of dataframe years 2022-2024
quali_list = [get_quali_times(y) for y in years]
# pd.concat combines the three dataframes into one
quali_df = pd.concat(quali_list, ignore_index=True) #ignore_index prevents duplicate numbering

#same thing, but for race results
race_list = [get_race_results(y) for y in years]
race_df = pd.concat(race_list, ignore_index=True)



## Encode drivers and constructors
le_driver = LabelEncoder() # Make Label encoder for drivers: convert categorical variable into numbers for XGboost to process

quali_df['Driver_encoded'] = le_driver.fit_transform(quali_df['Driver'])
# Handle potential unseen drivers in race_df
race_df['Driver_encoded'] = race_df['Driver'].map(
    lambda d: le_driver.transform([d])[0] if d in le_driver.classes_ else -1
)
le_team = LabelEncoder() # label encoder for teams
race_df['Constructor_encoded'] = le_team.fit_transform(race_df['Constructor'])



## Train XGBoost to predict Qualifying Times

# Train test split
quali_train = quali_df[quali_df['Year'] < 2024]
quali_test = quali_df[quali_df['Year'] == 2024]

#find non-linear relationship between assigned x & y values
X_quali_train = quali_train[['Driver_encoded']]
y_quali_train = quali_train['LapTime_sec']

X_quali_test = quali_test[['Driver_encoded']]
y_quali_test = quali_test['LapTime_sec']

quali_model = xgb.XGBRegressor( # need to further tweak model parameters
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
quali_model.fit(X_quali_train, y_quali_train)
pred_quali_test = quali_model.predict(X_quali_test)

mae_quali = mean_absolute_error(y_quali_test, pred_quali_test)
print(f"Qualifying Time MAE (seconds): {mae_quali:.3f}")



## Merge predicted Quali times into race dataset
race_2024 = race_df[race_df['Year'] == 2024].copy()

# Match predicted quali times by driver
race_2024 = race_2024.merge(
    quali_test[['Driver', 'LapTime_sec']],
    on='Driver',
    how='left'
)
race_2024.rename(columns={'LapTime_sec': 'PredictedQualiTime'}, inplace=True)

## Also include past performances
# Find avg finishing position for 2022-2023
past_performance = race_df[race_df['Year'] < 2024].groupby('Driver')['Position'].mean().reset_index()
past_performance.rename(columns={'Position': 'AvgPastFinishing'}, inplace=True)

race_2024 = race_2024.merge(past_performance, on='Driver', how='left')

# Fill missing values
race_2024.fillna({
    'PredictedQualiTime': race_2024['PredictedQualiTime'].mean(),
    'AvgPastFinishing': race_2024['AvgPastFinishing'].mean(),
    'Driver_encoded': -1
}, inplace=True)


##Train XGBoost to predict Race Positions
race_features = ['Driver_encoded', 'Constructor_encoded', 'PredictedQualiTime', 'AvgPastFinishing']
X_race = race_2024[race_features]
y_race = race_2024['Position']

race_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

race_model.fit(X_race, y_race)
pred_race_positions = race_model.predict(X_race)

mae_race = mean_absolute_error(y_race, pred_race_positions)
print(f"Race Position MAE: {mae_race:.2f}")

## Add predictions to dataframe
race_2024['PredictedRacePos'] = pred_race_positions
print(race_2024[['Driver', 'Constructor', 'PredictedQualiTime', 'AvgPastFinishing', 'PredictedRacePos']])

 