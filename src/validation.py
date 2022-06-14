import os
import sys
# sys.path.append(os.path.abspath('../'))
from multiprocessing import Pool
from fbprophetGamma import Prophet as ProphetGamma
from fbprophetNB import Prophet as ProphetNB
from fbprophet import Prophet
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay, precision_score, recall_score
from utils.ignore_prophet_warnings import suppress_stdout_stderr
import yaml
import logging

logging.getLogger('fbprophet').setLevel(logging.WARNING)
USE_MULTIPROCESSING = True

DATA_PATH = 'data/processed/processed.parquet'
REPORT_FOLDER = 'reports/naive/'
OUTPUT_FILE = 'files/scores.csv'

MODELS_DICT = {'Prophet': Prophet, 'ProphetGamma': ProphetGamma, 'ProphetNB': ProphetNB}
SALE_DATES = ['2021-09-14', '2020-06-23', '2020-09-15', '2020-11-10',
              '2020-12-15', '2021-01-19', '2021-03-23']

if os.path.exists(DATA_PATH):
    tbl = pd.read_parquet(DATA_PATH)
else:
    os.system('python3 utils/preprocessing.py')
    tbl = pd.read_parquet(DATA_PATH)


def make_sales(sale_list=SALE_DATES):
    sales = pd.DataFrame({
            'holiday': 'sale',
            'ds': pd.to_datetime(sale_list),
            'lower_window': 0,
            'upper_window': 4,
        })
    return sales


def make_interval(model, data, interval_width):
    size = 10000
    q = (1 - interval_width) / 2
    mean = np.mean(data.yhat)
    std = np.std(data.y - data.yhat)
    var = np.var(data.y - data.yhat)
    
    yhat_lower, yhat_upper = [], []
    if model == Prophet:
        for x in data.yhat:
            dist = np.random.normal(x, std, size)
            yhat_lower.append(np.quantile(dist, q))
            yhat_upper.append(np.quantile(dist, 1 - q))
    elif model == ProphetNB:
        for x in data.yhat:
            p = x / var
            n = x ** 2 / (var - x)
            dist = np.random.negative_binomial(n, p, size)
            yhat_lower.append(np.quantile(dist, q))
            yhat_upper.append(np.quantile(dist, 1 - q))
    elif model == ProphetGamma:
        for x in data.yhat:
            theta = var / x
            k = x / theta
            dist = np.random.gamma(k, theta, size)
            yhat_lower.append(np.quantile(dist, q))
            yhat_upper.append(np.quantile(dist, 1 - q))
    else:
        NotImplementedError('Model could be only [Prophet, ProphetNB, ProphetGamma]')
        
    data['yhat_lower'] = yhat_lower
    data['yhat_upper'] = yhat_upper
    return data


def validate_anomalies(model, target, anom_params=(0.2, 3, True),
                       test_time_series_duration=len(tbl), to_plot=False, 
                       use_random_init_time=True, use_rolling_window=True, use_naive_sol=True):
    '''
    Return true and predicted by model indexes of anomalies
    model: Prophet or ProphetGamma
    data.columns = ['ds', 'y']
    '''
    anom_coeff, num_anomalies, in_a_row = anom_params

    if model == Prophet:
        interval_width = PROPHET_WIDTH
    elif model == ProphetNB:
        interval_width = NB_WIDTH
    elif model == ProphetGamma:
        interval_width = GAMMA_WIDTH
    else:
        interval_width = 0.99
    data = tbl[[target]].reset_index().rename(columns={target: 'y', 'product_creation_date': 'ds'}).copy()

    if use_random_init_time:
        init = np.random.randint(0, 24)
        num_hours = 24 * (1 - anom_coeff)
        init_coeff = 1 - (24 - init) / 24 if (24 - init - num_hours) < 0 else anom_coeff
        extra_hours = abs(24 - init - num_hours) if (24 - init - num_hours) < 0 else 0
        extra_coeff = 1 - extra_hours / 24
    else:
        init_coeff = anom_coeff
        extra_coeff = 1

    if in_a_row:
        start = np.random.randint(test_time_series_duration - num_anomalies - 2)
        art_anomalies = list(np.arange(start, start + num_anomalies))
    else:
        art_anomalies = np.random.choice(test_time_series_duration, num_anomalies, replace=False)
        art_anomalies = list(sorted(art_anomalies))
    art_anomalies.append(art_anomalies[-1] + 1)

    test_time_series = data.y.tolist()[-test_time_series_duration:]

    for i, ind in enumerate(art_anomalies):
        if i == 0:
            test_time_series[ind] = init_coeff * test_time_series[ind]
        elif i == len(art_anomalies) - 1:
            test_time_series[ind] = extra_coeff * test_time_series[ind]
        else:
            test_time_series[ind] = anom_coeff * test_time_series[ind]

    data['y'] = data.y.tolist()[:-test_time_series_duration] + test_time_series
    if use_rolling_window:
        data['y'] = data.y.rolling(window=num_anomalies, min_periods=1, center=True).mean()#.round()
        
    if use_naive_sol:
        preds = np.where(data.y[-test_time_series_duration:] <= 1e-3)[0]
    else:
        data['y'] = data.y.apply(lambda x: 1e-1 if x < 1e-1 else x)
        data['cap'] = data.y.max()

        m = model(
            weekly_seasonality=True,
            # yearly_seasonality=True,
            holidays=sales,
            interval_width=interval_width,
            seasonality_mode='multiplicative'
        )
        try:
            with suppress_stdout_stderr():
                m.fit(data)
        except:
            print(f'There was a problem with model {model}, target {target}, and params {anom_coeff}, {num_anomalies}')
        forecast = m.predict(data)

        comp = pd.merge(
            forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']],
            data,
            left_on='ds',
            right_on='ds')

        # if use_rolling_window:
        #     comp = make_interval(model, comp, interval_width=interval_width)

        comp.y = comp.y.apply(lambda x: 1e-3 if x <= 1e-1 else x)
        anomalies = comp[(comp.y < comp.yhat_lower)]  # | (comp.y > comp.yhat_upper)]
        anomalies = anomalies[anomalies.ds >= data.ds[-test_time_series_duration:].values[0]]
        if len(anomalies) != 0:
            preds = [np.where(data.ds[-test_time_series_duration:] == x)[0][0] for x in anomalies.ds]
        else:
            preds = []

        if to_plot:
            plt.figure(figsize=(30, 9))
            ax = plt.gca()
            plt.title(f'{target}', fontsize=20)
            m.plot(forecast, ax=ax, plot_cap=False)
            ax.scatter(comp.ds[-test_time_series_duration:].iloc[art_anomalies],
                       comp.y[-test_time_series_duration:].iloc[art_anomalies], c='g', marker='o', s=80)

            r = data.ds[-test_time_series_duration:].iloc[art_anomalies].tolist()
            for i, x in anomalies.iterrows():
                if x.ds in r:
                    ax.scatter(x.ds, x.y, c='r', marker='o', s=80)
                else:
                    ax.scatter(x.ds, x.y, c='yellow', marker='o', s=80)
            new_dir_name = os.path.join(REPORT_FOLDER,
                        f'plots/interval_width={round(interval_width, 3)}/{target}_{num_anomalies}_{round(anom_coeff, 2)}')
            os.makedirs(new_dir_name, exist_ok=True)
            plt.savefig(os.path.join(new_dir_name, f'{model.__module__.split(".")[0]}.png'))
            plt.close()

    art_anomalies_fin = [int(x in art_anomalies) for x in range(test_time_series_duration)]
    preds = [int(x in preds) for x in range(test_time_series_duration)]
    
    
    if not use_random_init_time or extra_coeff == 1:
        art_anomalies_fin[art_anomalies[-1]] = 0
    else:
        if preds[art_anomalies[0]] == 1 or preds[art_anomalies[-1]] == 1:
            art_anomalies_fin[art_anomalies[0]], preds[art_anomalies[0]] = 1, 1
            art_anomalies_fin[art_anomalies[-1]], preds[art_anomalies[-1]] = 0, 0
        else:
            art_anomalies_fin[art_anomalies[0]], preds[art_anomalies[0]] = 1, 0
            art_anomalies_fin[art_anomalies[-1]], preds[art_anomalies[-1]] = 0, 0
            
    if num_anomalies > 1 and use_rolling_window:
        is_detected = 0
        for i in range(num_anomalies):
            if preds[art_anomalies[i]] == 1:
                is_detected = 1
                break
        for i in range(1, num_anomalies):
            art_anomalies_fin[art_anomalies[i]], preds[art_anomalies[i]] = 0, 0
        art_anomalies_fin[art_anomalies[0]], preds[art_anomalies[0]] = 1, is_detected
        
    return art_anomalies_fin, preds


def conduct_experiments(model, target, anom_params=(0.2, 3, True),
                        test_time_series_duration=len(tbl), n_experiments=100):
    '''Return confusion matrix for all experimenst'''
    anom_coeff, num_anomalies, in_a_row = anom_params
    actual = []
    preds = []
    if model == Prophet:
        interval_width = PROPHET_WIDTH
    elif model == ProphetNB:
        interval_width = NB_WIDTH
    elif model == ProphetGamma:
        interval_width = GAMMA_WIDTH
    else:
        interval_width = 0.99

    i = 0
    while i < n_experiments:
        try:
            # anomalies, predictions = validate_anomalies(model=model, target=target, anom_coeff=anom_coeff,
            #                                             num_anomalies=num_anomalies, in_a_row=in_a_row,
            #                                             test_time_series_duration=test_time_series_duration)
            anomalies, predictions = validate_anomalies(model=model, target=target, 
                                                        anom_params=(anom_coeff, num_anomalies, in_a_row),
                                                        test_time_series_duration=test_time_series_duration)
            i += 1
        except:
            continue
        actual += anomalies
        preds += predictions
    f1 = f1_score(actual, preds)
    precision = precision_score(actual, preds)
    recall = recall_score(actual, preds)
    report = pd.DataFrame(
        [[model.__module__.split(".")[0], str(target), n_experiments, anom_coeff, num_anomalies, in_a_row, interval_width, f1, precision, recall]],
        columns=['model', 'target_col', 'n_experiments', 'anom_coeff', 'num_anomalies', 'in_a_row', 'interval_width', 'f1_score', 'precision', 'recall'])
    if os.path.exists(os.path.join(REPORT_FOLDER, OUTPUT_FILE)):
        report.to_csv(os.path.join(REPORT_FOLDER, OUTPUT_FILE), mode='a', header=False, index=False)
    else:
        report.to_csv(os.path.join(REPORT_FOLDER, OUTPUT_FILE), index=False)
    cm = confusion_matrix(actual, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.grid()
    new_dir_name = os.path.join(REPORT_FOLDER,
                                f'plots/interval_width={round(interval_width, 3)}/{target}_{num_anomalies}_{round(anom_coeff, 2)}')
    os.makedirs(new_dir_name, exist_ok=True)
    plt.savefig(os.path.join(new_dir_name, f'{model.__module__.split(".")[0]}_cm.png'))
    plt.close()


if __name__ == '__main__':
    sales = make_sales()

    with open('configs/config_interval_width.yaml', 'r') as f:
        config_interval_width = yaml.load(f)

    PROPHET_WIDTH = config_interval_width['Prophet']
    GAMMA_WIDTH = config_interval_width['ProphetGamma']
    NB_WIDTH = config_interval_width['ProphetNB']

    with open('configs/config.yaml', 'r') as f:
        config = yaml.load(f)
    models = [MODELS_DICT[key] for key in config['models']]
    
    if USE_MULTIPROCESSING:
        if config['mode'] == 1:
            args = product(models, config['columns'], [list(x.values()) for x in config['scenarios']],
                           [len(tbl)], [True], [True])
            with Pool(32) as p:
                x = p.starmap(validate_anomalies, args)
        elif config['mode'] == 2:
            args = product(models, config['columns'], [list(x.values()) for x in config['scenarios']],
                           [len(tbl)], [800])
            with Pool(32) as p:
                x = p.starmap(conduct_experiments, args)
        else:
            raise NotImplementedError(
                f'В config.yaml нужно выставить config.mode равным 1 или 2, сейчас: {config["mode"]}')
    else:
        for target in config['columns']:
            for params in config['scenarios']:
                anom_params = (params['anom_coeff'], params['num_anomalies'], params['in_a_row'])
                for m in models:
                    if config['mode'] == 1:
                        x = validate_anomalies(m, target, to_plot=True, anom_params=anom_params)
                    elif config['mode'] == 2:
                        conduct_experiments(m, target, n_experiments=50, anom_params=anom_params)
                    else:
                        raise NotImplementedError(
                        f'В config.yaml нужно выставить config.mode равным 1 или 2, сейчас: {config["mode"]}')


