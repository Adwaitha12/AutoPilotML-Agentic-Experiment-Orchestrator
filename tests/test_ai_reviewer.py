import pandas as pd
from agents.data_engineer import DataEngineer
from agents.ml_strategy import MLStrategy
from agents.model_architect import ModelArchitect
from agents.experiment_agent import ExperimentAgent
from agents.performance_analyst import PerformanceAnalyst
from agents.ai_reviewer import AIReviewer


def run_test():
    df = pd.read_csv('datasets/sample_loan_data.csv')

    # Clean
    de = DataEngineer()
    X_clean, y, preproc = de.clean(df, target='Loan_Approved')

    # Strategy
    strategy = MLStrategy()
    task = strategy.determine_task(df, 'Loan_Approved')['task_type']

    # Models
    ma = ModelArchitect()
    models = ma.select_models(task)

    # Experiment
    ea = ExperimentAgent(output_dir='outputs/models_ai_review')
    results = ea.train_models(models, X_clean, y, task)

    # Performance analysis
    pa = PerformanceAnalyst()
    analysis = pa.analyze(results, task)

    # Review
    reviewer = AIReviewer()
    review = reviewer.review(df, 'Loan_Approved', results)

    print('Findings:')
    for f in review['findings']:
        print('-', f)
    print('\nRecommendations:')
    for r in review['recommendations']:
        print('-', r)


if __name__ == '__main__':
    run_test()
