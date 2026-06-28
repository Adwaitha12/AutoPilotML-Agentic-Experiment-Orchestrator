from agents.experiment_agent import ExperimentAgent
from agents.data_engineer import DataEngineer
from agents.model_architect import ModelArchitect
from agents.ml_strategy import MLStrategy
from agents.performance_analyst import PerformanceAnalyst
import pandas as pd


def run_test():
    df = pd.read_csv('datasets/sample_loan_data.csv')

    # Prepare data
    de = DataEngineer()
    X_clean, y, preproc = de.clean(df, target='Loan_Approved')

    # Determine task
    strategy = MLStrategy()
    task = strategy.determine_task(df, 'Loan_Approved')['task_type']

    # Select models
    ma = ModelArchitect()
    models = ma.select_models(task)

    # Run experiment
    agent = ExperimentAgent(output_dir='outputs/models_pa_test')
    results = agent.train_models(models, X_clean, y, task)

    # Analyze
    pa = PerformanceAnalyst()
    analysis = pa.analyze(results, task)

    print('Recommendation:')
    print(analysis['recommendation'])
    print('\nTop models:')
    for m in analysis['leaderboard']:
        print(m['name'], m.get('metrics'))


if __name__ == '__main__':
    run_test()
