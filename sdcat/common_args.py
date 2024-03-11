# sightwire, Apache-2.0 license
# Filename: common_args.py
# Description: Common arguments for processing commands

import click

from sdcat.config.config import default_config_ini

# Common arguments for processing commands
config_ini = click.option('--config-ini',
                          type=str,
                          default=default_config_ini,
                          help=f'Path to config file to override. Defaults are in {default_config_ini}. Copy to your own custom.ini file to override')