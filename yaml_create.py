import os
import yaml

yml = {
  'MODEL': 'CNN49u',

  'DATASET':{
    'LOOKBACK_WIN': 49,
    'START_DATE': 20211101,
    'END_DATE': 20211110,
    'MODE': 'train',
    'INDICATORS': {'MA': {'WIN': 49 * 5}}, # Could be None
    'SHOW_VOLUME': False, # using data augmentation
    'SAMPLE_RATE': 0.2,
    'PARALLEL_NUM': 12
  },

  'PATHS': {
    'COMMONCACHE_DIR': '/path/to/commoncache',
    'IMAGE_DATA_DIR': '/path/to/image/data',
    'PROJECT_DIR': '/path/to/project/output',
  },

  'DATA_FILES': {
    'OPEN':          'path/to/open_data',
    'HIGH':          'path/to/high_data',
    'LOW':           'path/to/low_data',
    'CLOSE':         'path/to/close_data',
    'VOLUME':        'path/to/volume_data',
    'ASHARE_FILTER': 'path/to/ashare_filter',
    'ADJ_FACTOR':    'path/to/adj_factor',
    'UP_LIMIT':      'path/to/up_limit',
    'DN_LIMIT':      'path/to/dn_limit',
    'TRADE_DATES':   'path/to/trade_dates',
    'UNIVERSE_UID':  'path/to/universe_uid',
  },

  'TRAIN':{
    'PREDICT_WIN': 50,
    'LABEL': 'RET_LONG',
    'START_DATE': 20170101,
    'END_DATE': 20201231,
    'VALID_RATIO': 0.06,
    'BATCH_SIZE': 256,
    'NEPOCH': 20,
    'LEARNING_RATE': 0.05,
    'WEIGHT_DECAY': 0.01,
    'MODEL_SAVE_FILE': '/path/to/models/I49R50_OHLC/I49R50_OHLC_1701-2012_Ed_1_0_0.tar',
    'LOG_SAVE_FILE': '/path/to/logs/I49R50_OHLC/I49R50_OHLC_1701-2012_Ed_1_0_0.csv',
    'EARLY_STOP_EPOCH': 5,
    'EDITION': 'Edition_1_0_0',
    'LR_BASE_RATE': 0.001,
    'WARMUP_EPOCH': 1
  },

  'INFERENCE':{
    'START_DATE': 20210101,
    'END_DATE': 20230630,
    'FACTORS_SAVE_FILE': '/path/to/factors/I49R50_OHLC/I49R50_OHLC_2101-2306_Ed_1_0_0.csv'
  }
}

yaml_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(yaml_dir, 'w') as c:
  yaml.dump(yml, c)