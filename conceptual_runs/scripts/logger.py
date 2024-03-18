import logging


logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("logger.log"), logging.StreamHandler()],
    format="%(name)s %(asctime)s %(levelname)s %(message)s",
)
