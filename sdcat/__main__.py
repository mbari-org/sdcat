# sdcat, Apache-2.0 license
# Filename: __main__.py
# Description: Main entry point for the sdcat command line interface
from datetime import datetime
from pathlib import Path

import click
from sdcat.logger import err, info, create_logger_file
from sdcat import __version__
from sdcat.cluster.commands import run_cluster
from sdcat.detect.commands import run_detect


create_logger_file(log_path=Path.home() / 'sdcat' / 'logs')
default_data_path = Path(__file__).parent / 'testdata'
default_model = 'MBARI/megabenthic'

@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(
    __version__,
    '-V', '--version',
    message=f'%(prog)s, version %(version)s'
)
def cli():
    """
    Process images from a command line.
    """
    pass

cli.add_command(run_detect)
cli.add_command(run_cluster)


if __name__ == '__main__':
    try:
        start = datetime.utcnow()
        cli()
        end = datetime.utcnow()
        info(f'Done. Elapsed time: {end - start} seconds')
    except Exception as e:
        err(f'Exiting. Error: {e}')
        exit(-1)