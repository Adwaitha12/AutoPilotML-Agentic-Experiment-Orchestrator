import pandas as pd
from agents.data_engineer import DataEngineer


def run_test():
    df = pd.read_csv('datasets/sample_loan_data.csv')
    de = DataEngineer()
    X_clean, y, preproc = de.clean(df, target='Loan_Approved')

    print('Original shape:', df.shape)
    print('Cleaned X shape:', X_clean.shape)
    print('Target present:', y is not None)
    print('Numeric features:', de.numeric_features)
    print('Categorical features:', de.categorical_features)
    print('First 5 rows of cleaned X:')
    print(X_clean.head())


if __name__ == '__main__':
    run_test()
