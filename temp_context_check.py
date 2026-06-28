from core.context import ExperimentContext

context = ExperimentContext(dataset=[{'a': 1}], target_column='a')
context.quality_report = {'status': 'ok'}
context.preprocessing_summary = {'steps': []}
context.selected_models = ['model']
context.trained_models = {'model': object()}
context.predictions = {'model': {'test_predictions': [1]}}
context.metrics = {'model': {'accuracy': 1.0}}
context.leaderboard = [{'model_name': 'model'}]
context.best_model = object()
context.critic_analysis = {'summary': 'ok'}
context.agent_thoughts = {'qa': 'ok'}
context.visualizations = {'chart': 'path'}
context.report_path = 'report.html'
context.execution_history = [{'agent': 'qa'}]
context.experiment_metadata = {'run': True}

print(sorted(context.to_dict().keys()))
print(context.clean_dataframe is None)
print(context.cleaned_dataframe is None)
