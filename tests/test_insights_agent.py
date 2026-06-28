import pandas as pd
from agents.data_engineer import DataEngineer
from agents.ml_strategy import MLStrategy
from agents.model_architect import ModelArchitect
from agents.experiment_agent import ExperimentAgent
from agents.insights_agent import InsightsAgent


def run_test():
    df = pd.read_csv('datasets/sample_loan_data.csv')

    # Data engineer
    de = DataEngineer()
    X_clean, y, preproc = de.clean(df, target='Loan_Approved')

    # Strategy
    strategy = MLStrategy()
    task = strategy.determine_task(df, 'Loan_Approved')['task_type']

    # Model selection
    ma = ModelArchitect()
    models = ma.select_models(task)

    # Run experiment and save to outputs
    ea = ExperimentAgent(output_dir='outputs/models_insights_test')
    results = ea.train_models(models, X_clean, y, task)

    # Generate insights
    ia = InsightsAgent(output_dir='outputs/charts_insights')
    summary = ia.generate_insights(df, X_clean, y, results, task)

    print('Generated artifacts:')
    for k, v in summary['generated'].items():
        print(k, v)


if __name__ == '__main__':
    run_test()
