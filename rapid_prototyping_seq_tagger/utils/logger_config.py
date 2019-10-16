
import os
import datetime

import logging
import logging.config

def load_logger_config():
    log_file_name = 'log/' + datetime.datetime.today().strftime('%Y%m%d-%H:%M') + '-' + os.uname().nodename +'.log'
    logger_config_dict = {
        'version': 1,

        'formatters': {
            'customStreamFormat': {
                'format': '%(message)s'
            },
            'customFileFormat': {
                'format': '%(asctime)s %(name)s.%(module)s.%(funcName)s:%(lineno)s %(levelname)s \t %(message)s'
            }
        },

        'handlers': {
            'customStreamHandler': {
                'class': 'logging.StreamHandler',
                'formatter': 'customStreamFormat',
                'level': logging.INFO
            },
            'customFileHandler': {
                'class': 'logging.FileHandler',
                'formatter': 'customFileFormat',
                'filename': log_file_name,
                'mode': 'w',
                'level': logging.DEBUG
            }
        },

        # ロガーの対象一覧
        'loggers': {
            '': {
                'handlers': ['customStreamHandler', 'customFileHandler'],
                'level': logging.DEBUG,
                'propagate': 0
            }
        }
    }

    logging.config.dictConfig(logger_config_dict)
    return
