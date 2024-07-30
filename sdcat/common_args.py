# sdcat, Apache-2.0 license
# Filename: common_args.py
# Description: Common arguments for processing commands

import click

from sdcat.config.config import default_config_ini

# Common arguments for processing commands
config_ini = click.option('--config-ini',
                          type=str,
                          default=default_config_ini,
                          help=f'Path to config file to override. Defaults are in {default_config_ini}. Copy to your own custom.ini file to override')

start_image = click.option('--start-image',
                            type=str,
                            help='Start image name')

end_image = click.option('--end-image',
                            type=str,
                            help='End image name')

alpha = click.option('--alpha',
                     type=float,
                     help='Alpha is a parameter that controls the linkage. See https://hdbscan.readthedocs.io/en/latest/parameter_selection.html. '
                          'Default is 0.92. Increase for less conservative clustering, e.g. 1.0')

cluster_selection_epsilon = click.option('--cluster-selection-epsilon',
                                         type=float,
                                         help='Epsilon is a parameter that controls the linkage. '
                                              'Default is 0. Increase for less conservative clustering')

cluster_selection_method = click.option('--cluster-selection-method',
                                        type=str,
                                        default='leaf',
                                        help='Method for selecting the optimal number of clusters. '
                                             'Default is leaf. Options are leaf, eom, and dill')

min_cluster_size = click.option('--min-cluster-size',
                                type=int,
                                help='The minimum number of samples in a group for that group to be considered a cluster. '
                                     'Default is 2. Increase for less conservative clustering, e.g. 5, 15')

use_tsne = click.option('--use-tsne',
                        is_flag=True,
                        help='Use t-SNE for dimensionality reduction. Default is False')