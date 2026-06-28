from agents.knowledge_base import KnowledgeBase


def run_test():
    kb = KnowledgeBase(db_path='database/experiments_test.db')

    sample = {
        'dataset_summary': {'rows': 100, 'columns': 5},
        'target': 'y',
        'task_type': 'classification',
        'models': [{'name': 'Logistic Regression'}],
        'best_model': {'name': 'Logistic Regression', 'metrics': {'accuracy': 0.9}},
        'metrics': {'accuracy': 0.9},
        'report_paths': {'html': 'path/to/report.html'}
    }

    eid = kb.save_experiment(
        name='test_experiment',
        dataset_summary=sample['dataset_summary'],
        target=sample['target'],
        task_type=sample['task_type'],
        models=sample['models'],
        best_model=sample['best_model'],
        metrics=sample['metrics'],
        report_paths=sample['report_paths'],
    )

    print('Saved experiment id:', eid)

    rows = kb.list_experiments(limit=10)
    print('Recent experiments:', rows)

    fetched = kb.get_experiment(eid)
    print('Fetched experiment:', fetched)


if __name__ == '__main__':
    run_test()
