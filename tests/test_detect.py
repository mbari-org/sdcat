# sdcat, Apache-2.0 license
# Filename: sdcat/tests/test_detect.py
# Description:    Test the detect command.

import tempfile
from pathlib import Path
import subprocess


def run_detect(data_dir: Path, scale: int) -> int:
    """Test the detect command.
    :param data_dir: The directory containing the test data.
    :param scale: The scale to use for detection.
    :return: The number of detections.
    """

    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent

    num_detections = 0
    # Run in temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run sdcat
        proc = subprocess.Popen(['python3',
                                 f'{root_dir}/sdcat/__main__.py',
                                 'detect',
                                 '--skip-sahi',
                                 '--scale-percent', str(scale),
                                 '--save-dir',
                                 tmp_dir,
                                 '--image-dir',
                                 data_dir.as_posix()], stdout=subprocess.PIPE)

        # Wait for the process to finish
        proc.wait()

        # Verify that the process finished successfully
        assert proc.returncode == 0

        # The output should have a total of num_detections lines, including 1 for the header
        # Data is filtered after detection and put in the det_filtered directory
        out_path = Path(tmp_dir) / 'det_filtered'
        for file in out_path.rglob('*.csv'):
            with open(file) as f:
                lines = f.readlines()
                num_detections = len(lines) - 1
                break

        return num_detections

def test_bird():
    """ Test that sdcat can detect the correct number of targets in a drone image with birds"""

    data_path = Path(__file__).parent / 'data' / 'bird'
    num_detections = run_detect(data_path, 25)
    print(f'Found {num_detections} in test_bird')
    assert num_detections == 20


def test_pinniped():
    """ Test that sdcat can detect the correct number of targets in a drone images with pinnipeds and waves"""

    data_path = Path(__file__).parent / 'data' / 'pinniped'
    num_detections = run_detect(data_path, 60)
    print(f'Found {num_detections} in test_pinniped')
    assert num_detections == 62


def test_plankton():
    """ Test that sdcat can detect the correct number of targets in an image with plankton"""

    data_path = Path(__file__).parent / 'data' / 'plankton'
    num_detections = run_detect(data_path, 80)
    print(f'Found {num_detections} in test_plankton')
    assert num_detections == 1013


if __name__ == '__main__':
    # test_bird()
    # test_pinniped()
    test_plankton()
    print('All tests passed')