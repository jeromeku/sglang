# tty_pdb.py
from pdb import Pdb

class TtyPdb(Pdb):
    def __init__(self):
        # Always use the controlling terminal
        self._tty_in = open("/dev/tty", "r")
        self._tty_out = open("/dev/tty", "w")
        super().__init__(stdin=self._tty_in, stdout=self._tty_out)

def set_trace():
    TtyPdb().set_trace()
