# -*- coding: utf-8 -*-
import logging

logger = logging.Logger(__name__)

def main():
    pass

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
 
    main()
