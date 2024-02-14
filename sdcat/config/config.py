# sdcat, Apache-2.0 license
# Filename: sdcat/config/config.py
# Description:   Configuration helper to setup defaults for sdcat

from configparser import ConfigParser
import os
from sdcat.logger import info

default_config_ini = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')


class Config:

    def __init__(self, path: str = None, quiet: bool = False):
        """
        Read the .ini file and parse it
        """
        self.parser = ConfigParser()
        if path:
            self.path = path
        else:
            self.path = default_config_ini

        if not os.path.isfile(self.path):
            info(f'Bad path to {self.path}. Is your {self.path} missing?')
            raise Exception(f'Bad path to {self.path}. Is your {self.path} missing?')

        self.parser.read(self.path)
        lines = open(self.path).readlines()
        if not quiet:
            info(f"=============== Config file {self.path} =================")
            for l in lines:
                info(l.strip())

            if not path:
                info(f"============ You can override these settings by creating a customconfig.ini file and pass that "
                     f"in with --config=customconfig.ini =====")

    def __call__(self, *args, **kwargs):
        assert len(args) == 2
        return self.parser.get(args[0], args[1])

    def save(self, *args, **kwargs):
        assert len(args) == 3
        self.parser.set(section=args[0], option=args[1], value=args[2])
        with open(self.path, 'w') as fp:
            self.parser.write(fp)
