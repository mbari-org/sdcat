# sdcat, Apache-2.0 license
# Filename: __main__.py
# Description: Main entry point for the sdcat command line interface
from datetime import datetime
from pathlib import Path

import click
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sdcat.logger import err, info, create_logger_file
from sdcat import __version__
from sdcat.cluster.commands import run_cluster_det, run_cluster_roi
from sdcat.detect.commands import run_detect


create_logger_file("sdcat")
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
    Process images either to detect or cluster similar objects from a command line.
    """
    pass

cli.add_command(run_detect)


@cli.group(name="cluster")
def cli_cluster():
    """
    Commands related to clustering images
    """
    pass


cli.add_command(cli_cluster)
cli_cluster.add_command(run_cluster_det)
cli_cluster.add_command(run_cluster_roi)


if __name__ == '__main__':
    try:
        start = datetime.utcnow()
        cli()
        end = datetime.utcnow()
        info(f'Done. Elapsed time: {end - start} seconds')
    except Exception as e:
        err(f'Exiting. Error: {e}')
        exit(-1)
