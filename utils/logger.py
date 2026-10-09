import sys


class Logger(object):
    def __init__(self, file_name='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
