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
                                        help='Method for selecting the optimal number of clusters. '
                                             'Default is leaf. Options are leaf, eom, and dill')

algorithm = click.option('--algorithm',
                            type=str,
                            help='Algorithm for clustering. Default is best.  bes, generic, prims_kdtree, boruvka_kdtree')

min_cluster_size = click.option('--min-cluster-size',
                                type=int,
                                help='The minimum number of samples in a group for that group to be considered a cluster. '
                                     'Default is 2. Increase for less conservative clustering, e.g. 5, 15')

min_sample_size = click.option('--min-sample-size',
                               type=int,
                               help='The number of samples in a neighborhood for a point to be considered as a core point. '
                                    'This includes the point itself. Default is 1. Increase for more conservative clustering')

vits_batch_size = click.option('--vits-batch-size',
                               type=int,
                               default=32,
                               help='Batch size for processing images. Default is 32')

hdbscan_batch_size = click.option('--hdbscan-batch-size',
                            type=int,
                            default=50000,
                            help='Batch size for HDBSCAN. Default is 50000. Increase for your available CPU/GPU memory. ')

use_pca = click.option('--use-pca',
                       is_flag=True,
                       help='Use PCA for dimensionality reduction before clustering. Default is False')

skip_visualization = click.option('--skip-visualization',
                                    is_flag=True,
                                    help='Skip visualization. Default is False')