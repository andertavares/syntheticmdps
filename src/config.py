import sys
import configparser


class Config(object):
    instance = None

    def __init__(self):
        self.instance = self
        # sets up some default values
        self.settings = {
            'ltd_type': 'gaussian',
            'team_sizes': [10],
            'bandit_sizes': [75],
            'mus': [0.2, 0.4],
            'upper_bounds': [0.25,0.75],
            'sigmas': [0.2],
            'trials': 1000,
            'executions': 10,
            'max_parallel_setup': 10,
            'max_parallel_runs': 5,
            'output_dir': 'results'
        }

    @staticmethod
    def get_instance():
        if Config.instance is None:
            Config.instance = Config()
        return Config.instance

    def parse(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        experiment = config['experiment']
        settings = dict()

        # retrieving values, observing defaults
        settings['ltd_type']    = experiment.get('ltd_type', 'gaussian')
        settings['team_sizes']  = experiment.get('team_sizes', '10,20')
        settings['bandit_sizes'] = experiment.get('bandit_sizes', "200,500")
        settings['mus']         = experiment.get('mus', "0.2,0.4,0.6")
        settings['upper_bounds'] = experiment.get('upper_bounds', "0.25,0.5,0.75")
        settings['sigmas']      = experiment.get('sigmas', '0.2')
        settings['trials']      = int(experiment.get('trials', 10000))
        settings['executions']  = int(experiment.get('executions', 1000))
        settings['max_parallel_setup'] = int(experiment.get('max_parallel_setup', 10))
        settings['max_parallel_runs'] = int(experiment.get('max_parallel_runs', 5))
        settings['output_dir'] = experiment.get('output_dir', 'results')

        # process 'list' types
        settings['team_sizes'] = [int(x) for x in settings['team_sizes'].split(',')]
        settings['bandit_sizes'] = [int(x) for x in settings['bandit_sizes'].split(',')]
        settings['mus'] = [float(x) for x in settings['mus'].split(',')]
        settings['upper_bounds'] = [float(x) for x in settings['upper_bounds'].split(',')]
        settings['sigmas'] = [float(x) for x in settings['sigmas'].split(',')]

        self.settings = settings
        return settings

#TODO write a test
