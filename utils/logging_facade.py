import logging
import sys


class LogFacade:
    @staticmethod
    def get_logger():
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        root.addHandler(ch)
        return root
