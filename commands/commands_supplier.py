from commands.algorithm_commands import NormalizingCommand, KMeansCommand


class CommandsSupplier:
    def __init__(self, logger):
        self.logger = logger

    def supply(self):
        return [NormalizingCommand(self.logger),
                KMeansCommand(self.logger)]
