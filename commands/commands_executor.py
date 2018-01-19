from functools import reduce


class CommandsExecutor:
    def __init__(self, commands_supplier, data_supplier):
        self.commands_supplier = commands_supplier
        self.data_supplier = data_supplier

    def execute(self):
        return reduce(lambda left, right: right.execute(left), self.commands_supplier.supply(),
                      self.data_supplier.supply())
