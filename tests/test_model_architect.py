from agents.model_architect import ModelArchitect


def run_test():
    ma = ModelArchitect()
    print('Classification candidates:')
    for spec in ma.select_models('classification'):
        print('-', spec['name'], type(spec['estimator']))

    print('\nRegression candidates:')
    for spec in ma.select_models('regression'):
        print('-', spec['name'], type(spec['estimator']))


if __name__ == '__main__':
    run_test()
