import pandas as pd
from core.context import ExperimentContext
from agents.qa_agent import DataQualityAgent
from agents.cleaning_agent import DataCleaningAgent
from agents.task_agent import TaskDetectionAgent
from agents.model_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.critic_agent import CriticAgent
from agents.visualization_agent import VisualizationAgent
from agents.report_agent import ReportAgent
from agents.memory_agent import MemoryAgent
from agents.planner import PlannerAgent

frame = pd.DataFrame({'feature1':[1,2,3,4,5,6], 'feature2':[2,3,4,5,6,7], 'target':[0,1,0,1,0,1]})
context = ExperimentContext(dataset=frame, target_column='target')
planner = PlannerAgent(context=context, agents={
    'quality': DataQualityAgent(),
    'cleaning': DataCleaningAgent(),
    'task': TaskDetectionAgent(),
    'models': ModelSelectionAgent(),
    'training': TrainingAgent(),
    'evaluation': EvaluationAgent(),
    'critic': CriticAgent(),
    'visualization': VisualizationAgent(output_dir='outputs/charts'),
    'report': ReportAgent(output_dir='outputs/reports'),
    'memory': MemoryAgent(db_path='outputs/experiments.db'),
})
planner.run(['quality','cleaning','task','models','training','evaluation','critic','visualization','report','memory'])
print('smoke_ok', context.problem_type, context.best_model is not None, bool(context.report_path))
