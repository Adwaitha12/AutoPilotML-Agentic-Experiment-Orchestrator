import pandas as pd
from agents.ml_strategy import MLStrategy


def run_test():
    df = pd.read_csv('datasets/sample_loan_data.csv')
    strategy = MLStrategy()

    print('Testing target: Loan_Approved')
    result = strategy.determine_task(df, 'Loan_Approved')
    print(result)

    print('\nTesting target: Income')
    result2 = strategy.determine_task(df, 'Income')
    print(result2)


if __name__ == '__main__':
    run_test()
