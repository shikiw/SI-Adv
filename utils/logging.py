# -*- coding: utf-8 -*-

import os
import sys


class Logging_str(object):
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path

    def write(self, str = None):
        assert str is not None
        with open(self.logfile_path, "a") as file_object:
            msg = str
            file_object.write(msg+'\n')
        print(str)