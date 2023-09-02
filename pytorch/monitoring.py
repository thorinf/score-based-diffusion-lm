import logging


def get_initialised_logger(logfile_path='logfile.log', log_level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('[%(asctime)s: %(name)s: %(levelname)s] %(message)s')
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
