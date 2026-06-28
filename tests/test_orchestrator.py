import pandas as pd
from core.orchestrator import run_full_experiment


def run_test():
    df = pd.read_csv('datasets/sample_loan_data.csv')
    result = run_full_experiment(df, 'Loan_Approved', name='orchestrator_test')
    print('Experiment name:', result['experiment_name'])
    print('KB id:', result['kb_id'])
    print('Strategy:', result['strategy'])
    print('Top model from performance:', result['performance']['best_model']['name'])
    print('Report paths:', result['report_paths'])


if __name__ == '__main__':
    run_test()
