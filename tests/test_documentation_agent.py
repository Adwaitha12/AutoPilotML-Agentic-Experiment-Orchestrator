import pandas as pd
from agents.data_engineer import DataEngineer
from agents.ml_strategy import MLStrategy
from agents.model_architect import ModelArchitect
from agents.experiment_agent import ExperimentAgent
from agents.performance_analyst import PerformanceAnalyst
from agents.insights_agent import InsightsAgent
from agents.ai_reviewer import AIReviewer
from agents.documentation_agent import DocumentationAgent


def run_test():
    df = pd.read_csv('datasets/sample_loan_data.csv')

    de = DataEngineer()
    X_clean, y, preproc = de.clean(df, target='Loan_Approved')

    strategy = MLStrategy()
    task = strategy.determine_task(df, 'Loan_Approved')['task_type']

    ma = ModelArchitect()
    models = ma.select_models(task)

    ea = ExperimentAgent(output_dir='outputs/models_doc_test')
    results = ea.train_models(models, X_clean, y, task)

    pa = PerformanceAnalyst()
    analysis = pa.analyze(results, task)

    ia = InsightsAgent(output_dir='outputs/charts_doc_test')
    insights = ia.generate_insights(df, X_clean, y, results, task)

    reviewer = AIReviewer()
    review = reviewer.review(df, 'Loan_Approved', results)

    doc = DocumentationAgent(output_dir='outputs/reports_doc_test')
    report_paths = doc.generate_report(review, results, analysis, insights)

    print('Report outputs:')
    print(report_paths)


if __name__ == '__main__':
    run_test()
