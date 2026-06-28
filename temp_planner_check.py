from core.context import ExperimentContext
from agents.planner import PlannerAgent

context = ExperimentContext(dataset=[{'a': 1, 'b': 2}], target_column='a')
planner = PlannerAgent(context=context, agents={'noop': type('Noop', (), {'execute': lambda self, ctx: ctx})()})
result = planner.run(['noop'])
print(result.experiment_metadata['planner_status'])
print(result.experiment_metadata['execution_status']['noop']['status'])
print(len(result.experiment_metadata['execution_history']))
