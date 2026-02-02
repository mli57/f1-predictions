import os
import fastf1
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

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

## Pull data from api (train 2022-2024, test 2025)
TEST_YEAR = 2025  # if 2025 Singapore not in FastF1 yet, set to 2024
years = [2022, 2023, 2024, 2025] if TEST_YEAR == 2025 else [2022, 2023, 2024]

# build list of dataframe years
print("Loading qualifying data...")
quali_list = [get_quali_times(y) for y in years]
# pd.concat combines the three dataframes into one
quali_df = pd.concat(quali_list, ignore_index=True) #ignore_index prevents duplicate numbering

#same thing, but for race results
print("Loading race data...")
race_list = [get_race_results(y) for y in years]
race_df = pd.concat(race_list, ignore_index=True)
print("Data loaded.")

## Add constructor to quali (car performance matters for lap time)
driver_year_team = race_df[['Driver', 'Year', 'Constructor']].drop_duplicates()
quali_df = quali_df.merge(driver_year_team, on=['Driver', 'Year'], how='left')

## Driver's average past quali time at Singapore (strong predictor when available)
quali_df = quali_df.sort_values(['Driver', 'Year'])
def avg_past_quali(row):
    past = quali_df[(quali_df['Driver'] == row['Driver']) & (quali_df['Year'] < row['Year'])]['LapTime_sec']
    return past.mean() if len(past) > 0 else np.nan
quali_df['AvgPastQualiSingapore_sec'] = quali_df.apply(avg_past_quali, axis=1)
quali_df['AvgPastQualiSingapore_sec'] = quali_df['AvgPastQualiSingapore_sec'].fillna(quali_df['LapTime_sec'].mean())

## Encode drivers and constructors
le_driver = LabelEncoder() # Make Label encoder for drivers: convert categorical variable into numbers for XGboost to process
le_driver.fit(list(quali_df['Driver'].dropna().unique()) + list(race_df['Driver'].unique()))
quali_df['Driver_encoded'] = quali_df['Driver'].map(
    lambda d: le_driver.transform([d])[0] if d in le_driver.classes_ else -1
)
# Handle potential unseen drivers in race_df
race_df['Driver_encoded'] = race_df['Driver'].map(
    lambda d: le_driver.transform([d])[0] if d in le_driver.classes_ else -1
)
le_team = LabelEncoder() # label encoder for teams
le_team.fit(race_df['Constructor'].astype(str).unique())
race_df['Constructor_encoded'] = race_df['Constructor'].astype(str).map(
    lambda c: le_team.transform([c])[0] if c in le_team.classes_ else -1
).fillna(-1).astype(int)
quali_df['Constructor_encoded'] = quali_df['Constructor'].map(
    lambda c: le_team.transform([str(c)])[0] if pd.notna(c) and str(c) in le_team.classes_ else -1
)
# Fill missing constructor encoding (e.g. quali only driver)
quali_df['Constructor_encoded'] = quali_df['Constructor_encoded'].fillna(-1).astype(int)



## Train XGBoost to predict Qualifying Times

# Train test split: train 2022-2024, test 2025
quali_train = quali_df[quali_df['Year'] < TEST_YEAR]
quali_test = quali_df[quali_df['Year'] == TEST_YEAR]

# Features: driver, constructor (car), year, and past avg quali at Singapore (better than driver alone)
quali_feature_cols = ['Driver_encoded', 'Constructor_encoded', 'Year', 'AvgPastQualiSingapore_sec']
X_quali_train = quali_train[quali_feature_cols]
y_quali_train = quali_train['LapTime_sec']

X_quali_test = quali_test[quali_feature_cols]
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

# Validation: Random Forest and Regression (Ridge) on same quali task
rf_quali = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
rf_quali.fit(X_quali_train, y_quali_train)
mae_quali_rf = mean_absolute_error(y_quali_test, rf_quali.predict(X_quali_test))
print(f"Qualifying Time MAE - Random Forest (seconds): {mae_quali_rf:.3f}")

reg_quali = Ridge(alpha=1.0, random_state=42)
reg_quali.fit(X_quali_train, y_quali_train)
mae_quali_reg = mean_absolute_error(y_quali_test, reg_quali.predict(X_quali_test))
print(f"Qualifying Time MAE - Regression/Ridge (seconds): {mae_quali_reg:.3f}")



## Merge predicted Quali times into race dataset (test year = 2025)
race_test = race_df[race_df['Year'] == TEST_YEAR].copy()

# Use model-predicted quali times for test year (from quali model above)
quali_test_with_pred = quali_test[['Driver', 'LapTime_sec']].copy()
quali_test_with_pred.rename(columns={'LapTime_sec': 'PredictedQualiTime'}, inplace=True)
quali_test_with_pred['PredictedQualiTime'] = pred_quali_test  # overwrite with model prediction
race_test = race_test.merge(quali_test_with_pred[['Driver', 'PredictedQualiTime']], on='Driver', how='left')

## Also include past performances (avg finishing position in train years)
past_performance = race_df[race_df['Year'] < TEST_YEAR].groupby('Driver')['Position'].mean().reset_index()
past_performance.rename(columns={'Position': 'AvgPastFinishing'}, inplace=True)
race_test = race_test.merge(past_performance, on='Driver', how='left')

# Fill missing values first so rank is well-defined
race_test.fillna({
    'PredictedQualiTime': race_test['PredictedQualiTime'].mean(),
    'AvgPastFinishing': race_test['AvgPastFinishing'].mean(),
    'Driver_encoded': -1
}, inplace=True)
# Grid = starting position (strong predictor of finish). Test: use actual Grid if available, else predicted quali rank
race_test['PredictedQualiRank'] = race_test['PredictedQualiTime'].rank(method='min').astype(int)
if 'Grid' in race_test.columns:
    race_test['Grid_for_model'] = pd.to_numeric(race_test['Grid'], errors='coerce')
    race_test['Grid_for_model'] = race_test['Grid_for_model'].fillna(race_test['PredictedQualiRank'])
else:
    race_test['Grid_for_model'] = race_test['PredictedQualiRank']

## Build train set for race model (2022-2024: actual quali time + year-aware past finish)
race_train = race_df[race_df['Year'] < TEST_YEAR].copy()
race_train = race_train.merge(
    quali_df[['Driver', 'Year', 'LapTime_sec']].rename(columns={'LapTime_sec': 'QualiTime_sec'}),
    on=['Driver', 'Year'], how='left'
)
# Past avg finishing position (only years before current year)
def past_finish_for_row(row):
    past = race_df[(race_df['Driver'] == row['Driver']) & (race_df['Year'] < row['Year'])]['Position']
    return past.mean() if len(past) > 0 else np.nan
race_train['AvgPastFinishing'] = race_train.apply(past_finish_for_row, axis=1)
race_train['AvgPastFinishing'] = race_train['AvgPastFinishing'].fillna(race_train['Position'].mean())
race_train['QualiTime_sec'] = race_train['QualiTime_sec'].fillna(race_train.groupby('Year')['QualiTime_sec'].transform('mean'))
race_train = race_train.rename(columns={'QualiTime_sec': 'PredictedQualiTime'})
# Grid (starting position) is a strong predictor of finish position
race_train['Grid_for_model'] = pd.to_numeric(race_train['Grid'], errors='coerce')
race_train['Grid_for_model'] = race_train['Grid_for_model'].fillna(race_train['Grid_for_model'].median())

##Train XGBoost to predict Race Positions (train on 2022-2024, test on 2025)
race_features = ['Driver_encoded', 'Constructor_encoded', 'Grid_for_model', 'PredictedQualiTime', 'AvgPastFinishing']
# Ensure labels are valid (no NaN/inf): coerce to numeric and drop bad rows
race_train_clean = race_train.copy()
race_train_clean['Position'] = pd.to_numeric(race_train_clean['Position'], errors='coerce')
race_train_clean = race_train_clean.loc[np.isfinite(race_train_clean['Position'])].copy()
race_train_clean = race_train_clean.dropna(subset=race_features)
X_race_train = race_train_clean[race_features].fillna(0).astype(np.float64)
y_race_train = np.asarray(race_train_clean['Position'], dtype=np.float64)

race_test_clean = race_test.copy()
race_test_clean['Position'] = pd.to_numeric(race_test_clean['Position'], errors='coerce')
race_test_clean = race_test_clean.loc[np.isfinite(race_test_clean['Position'])].copy()
race_test_clean = race_test_clean.dropna(subset=race_features)
X_race_test = race_test_clean[race_features].fillna(0).astype(np.float64)
y_race_test = np.asarray(race_test_clean['Position'], dtype=np.float64)

race_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

race_model.fit(X_race_train, y_race_train)
pred_race_positions = race_model.predict(X_race_test)

mae_race = mean_absolute_error(y_race_test, pred_race_positions)
print(f"Race Position MAE (test {TEST_YEAR}): {mae_race:.2f}")

# Validation: Random Forest and Regression (Ridge) on same race task
rf_race = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
rf_race.fit(X_race_train, y_race_train)
mae_race_rf = mean_absolute_error(y_race_test, rf_race.predict(X_race_test))
print(f"Race Position MAE - Random Forest: {mae_race_rf:.2f}")

reg_race = Ridge(alpha=1.0, random_state=42)
reg_race.fit(X_race_train, y_race_train)
mae_race_reg = mean_absolute_error(y_race_test, reg_race.predict(X_race_test))
print(f"Race Position MAE - Regression/Ridge: {mae_race_reg:.2f}")

## Add predictions to dataframe
race_test_clean = race_test_clean.copy()
race_test_clean['PredictedRacePos'] = pred_race_positions
print(race_test_clean[['Driver', 'Constructor', 'PredictedQualiTime', 'AvgPastFinishing', 'PredictedRacePos', 'Position']])
