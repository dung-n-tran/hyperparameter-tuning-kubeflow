class Logger:
    def __init__(self, logger, type):
        """Logger wrapper

        Params:
            logger (object): Logger object
            type (str): Either 'aml' or 'python'
        """
        self.logger = logger
        self.type = type

    def log(self, metrics, value):
        if self.type == 'aml':
            self.logger.log(metrics, value)
        elif self.type == 'python':
            import logging
            logging.basicConfig(level=logging.INFO)
            self.logger.info("{}={}".format(metrics, value))
        elif self.type == 'katib':
            print("{}={}".format(metrics, value))
