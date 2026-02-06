import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    creates and configure a logger
    
    :param name: Description
    :type name: str
    :param log_file: Description
    :type log_file: str
    :param level: Description
    """

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger