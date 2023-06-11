# Dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb 
from sklearn.metrics import mean_squared_error

def read_File():
    # Read csv file

    df = pd.read_csv(r"C:\Users\Puteri Setyo Utami\Downloads\Train.csv\Train.csv")
    df = df.drop(['store','item'], axis=1)

    return df

def monthly_sales(df):
    # process dataframe to monthly sales
    
    monthly_df = df.copy()
    monthly_df.date = monthly_df.date.apply(lambda x: str(x)[:-3])
    monthly_df = monthly_df.groupby('date')['sales'].sum().reset_index()
    monthly_df.date = pd.to_datetime(monthly_df.date)
    monthly_df = monthly_df.set_index('date')
    monthly_df.index = pd.to_datetime(monthly_df.index)
    
    return monthly_df

def tts(df):
    # Split train and test data

    train = df.loc[df.index < '01-12-2017'] 
    test = df.loc[df.index >= '01-01-2017']
    
    return train, test

def feature(df):
    # Add additional features
    
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['weekofyear'] = df.index.isocalendar().week
    
    return df

def model(train,test):
    # Make the model

    FEATURES = ['year', 'month', 'day', 'dayofyear', 'dayofweek']
    TARGET = ['sales']

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                            n_estimators=4000,
                            early_stopping_rounds=50,
                            objective='reg:linear',
                            max_depth=3,
                            learning_rate=0.01)

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train),(X_test,y_test)],
            verbose=100)
    
    return X_train, y_train, X_test, y_test, reg

def forecast(df, test, X_test, reg):
# Forecasting
    test['prediction'] = reg.predict(X_test)
    df = df.merge(test[['prediction']],how='left',left_index=True,right_index=True)
    ax = df[['sales']].plot(figsize=(15,5))
    df['prediction'].plot(ax=ax, style='-')
    plt.legend(['Actual Data','Prediction Data'])
    plt.show()

def main():
    df = read_File()

    monthly = monthly_sales(df)

    train, test = tts(monthly)

    train = feature(train)
    test = feature(test)
    monthly = feature(monthly)

    X_train, y_train, X_test, y_test, reg = model(train, test)

    forecast(monthly,test,X_test, reg)

if __name__ == "__main__":
    main()