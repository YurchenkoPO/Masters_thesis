import os
import numpy as np
import yaml
from tqdm import tqdm

CONFIG_PATH = 'configs/config_interval_width.yaml'

STEP = 0.005
MIN_WIDTH = 0.6
MAX_WIDTH = 1


if __name__ == '__main__':
    for width in tqdm(np.arange(MIN_WIDTH, MAX_WIDTH + STEP, STEP)):
        width = float(round(width, 3))
        width_dict = {'Prophet': width,
                      'ProphetGamma': width,
                      'ProphetNB': width}

        with open(CONFIG_PATH, 'w') as outfile:
            yaml.dump(width_dict, outfile, default_flow_style=False)

        os.system('python3 src/validation.py')
