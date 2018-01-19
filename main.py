from commands.commands_executor import CommandsExecutor
from commands.commands_supplier import CommandsSupplier
from data.data_supplier import InputDataSupplier

# K = input('K: ')
# a = input('α: ')
from utils.logging_facade import LogFacade

logger = LogFacade.get_logger()
data_supplier = InputDataSupplier()
supplier = CommandsSupplier(logger)
executor = CommandsExecutor(supplier, data_supplier)
executor.execute()
