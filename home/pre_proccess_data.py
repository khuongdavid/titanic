import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import string
SEED = 42

def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)
def pre_proccess_data(passenger_dict):
    df = pd.DataFrame(passenger_dict)

    fare_bins = [0., 7.25, 7.75, 7.8958, 8.05, 10.5, 13., 15.7417, 23.25, 26.55, 34.07529231, 56.4958, 83.475, 512.3292]
    # Lấy giá trị Fare từ passenger_dict
    fare_value = passenger_dict['Fare'][0]

    # Tạo một Series từ giá trị Fare
    fare_series = pd.Series([fare_value])
    fare_labels = pd.cut(fare_series, bins=fare_bins, labels=range(len(fare_bins)-1), include_lowest=True).iloc[0]
    df['Fare'] = fare_labels

    age_bins =[ 0.17, 16.  , 21.  , 22.  , 25.  , 26.  , 29.5 , 34.  , 40.  ,
       48. , 80.  ]
    age_value = passenger_dict['Age'][0]
    age_series = pd.Series([age_value])
    age_labels = pd.cut(age_series, bins=age_bins, labels=range(len(age_bins)-1), include_lowest=True).iloc[0]
    df['Age'] = age_labels



    for i in range(1,4):
        if passenger_dict['Pclass'][0] == i:
            df[f'Pclass_{i}'] = 1
        else:
            df[f'Pclass_{i}'] = 0
    df.drop(['Pclass'], axis=1, inplace=True)
    for i in range(1,3):
        if passenger_dict['Sex'][0] == i:
            df[f'Sex_{i}'] = 1
        else:
            df[f'Sex_{i}'] = 0
    df.drop(['Sex'], axis=1, inplace=True)
    for i in range(1,5):
        if passenger_dict['Deck'][0] == i:
            df[f'Deck_{i}']= 1
        else:
            df[f'Deck_{i}'] = 0
    df.drop(['Deck'], axis=1, inplace=True)
    for i in range(1,4):
        if passenger_dict['Embarked'][0] == i:
            df[f'Embarked_{i}'] = 1
        else:
            df[f'Embarked_{i}'] = 0
    df.drop(['Embarked'], axis=1, inplace=True)
    for i in range(1,5):
        if passenger_dict['Title'][0] == i:
            df[f'Title_{i}'] = 1
        else:
            df[f'Title_{i}'] = 0
    df.drop(['Title'], axis=1, inplace=True)


    family_size_bins = [0, 1.5, 4.5, 6.5, 11.5]
    family_value = passenger_dict['Family_Size_Grouped'][0]
    family_series = pd.Series([family_value])
    family_labels = pd.cut(family_series, bins=family_size_bins, labels=range(len(family_size_bins)-1), include_lowest=True).iloc[0]
    

    for i in range(1,5):
        if family_labels == i:
            df[f'Family_Size_Grouped_{i}'] = 1
        else:
            df[f'Family_Size_Grouped_{i}'] = 0
    df.drop(['Family_Size_Grouped'], axis=1, inplace=True)

    df['Survival_Rate'] = 0.383838383838383
    df['Survival_Rate_NA'] = 0
    df.to_csv("helo.csv")
    df_test = pd.read_csv('df_test_temp.csv')
    df_test.drop(['PassengerId'], axis=1, inplace=True)
    df_all = concat_df(df, df_test)
    # Danh sách các cột theo thứ tự mong muốn
    desired_columns = ['Age', 'Fare', 'Ticket_Frequency', 'Is_Married', 
    'Survival_Rate', 'Survival_Rate_NA', 'Pclass_1', 'Pclass_2', 'Pclass_3', 
    'Sex_1', 'Sex_2', 'Deck_1', 'Deck_2', 'Deck_3', 'Deck_4', 'Embarked_1', 
    'Embarked_2', 'Embarked_3', 'Title_1', 'Title_2', 'Title_3', 'Title_4', 
    'Family_Size_Grouped_1', 'Family_Size_Grouped_2', 'Family_Size_Grouped_3', 
    'Family_Size_Grouped_4']

    # Sử dụng reindex để chọn lại các cột theo thứ tự mong muốn
    df_all = df_all.reindex(columns=desired_columns)

    X_test = StandardScaler().fit_transform(df_all)
    return X_test[0]

    
