import logging
import random

logging.basicConfig(filename="Sample.txt",
                    filemode='a',
                    format='%(asctime)s %(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Starting the loop for 15 iterations")
logging.info("hi")
