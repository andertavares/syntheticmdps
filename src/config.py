import sys
import configparser
'''

config['experiment'] = {
    'ltd_type': 'gaussian',
    'team_sizes': ,
    'bandit_sizes': ,
    'mus': ,
    'sigmas': '0.2',
    'trials': '10000',
    'executions': '1000'

}'''


def parse(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    settings = {}

    experiment = config['experiment']

    # retrieving values, observing defaults
    settings['ltd_type']    = experiment.get('ltd_type', 'gaussian')
    settings['team_sizes']  = experiment.get('team_sizes', '10,20')
    settings['bandit_sizes'] = experiment.get('bandit_sizes', "200,500")
    settings['mus']         = experiment.get('mus', "0.2,0.4,0.6")
    settings['sigmas']      = experiment.get('sigmas', '0.2')
    settings['trials']      = int(experiment.get('trials', 10000))
    settings['executions']  = int(experiment.get('executions', 1000))

    # process 'list' types
    settings['team_sizes'] = [int(x) for x in settings['team_sizes'].split(',')]
    settings['bandit_sizes'] = [int(x) for x in settings['bandit_sizes'].split(',')]
    settings['mus'] = [float(x) for x in settings['mus'].split(',')]
    settings['sigmas'] = [float(x) for x in settings['sigmas'].split(',')]

    return settings

#TODO write a test
