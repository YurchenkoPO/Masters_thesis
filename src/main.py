from src.validation import *


REPORT_FOLDER = 'reports/new/'

if os.path.exists(DATA_PATH):
    tbl = pd.read_parquet(DATA_PATH)
else:
    os.system('python3 utils/preprocessing.py')
    tbl = pd.read_parquet(DATA_PATH)


def time_series_classification(target_col):
    """
    TODO: find tresholds to classify data into 3 classes:
    1 - good (a lot of sales),
    2 - medium (common prophet can't handle anomalies)
    3 - bad (a lot of zeros in data)
    """
    if target_col in ['total', 'seat']:
        return 1
    elif target_col in ['gift_certificate']:
        return 2
    elif target_col in ["('AAQ-DME', 'auto_checkin')", "('DME-GDZ', 'auto_checkin')", "('DME-ORY', 'seat')"]:
        return 3
    else:
        raise NotImplementedError()


def validate_time_series(model, target, anom_coeff=0.2, num_anomalies=3, in_a_row=True,
                       test_time_series_duration=len(tbl), to_plot=False, use_random_init_time=True):
    '''
    Return true and predicted by model indexes of anomalies
    model: Prophet or ProphetGamma
    data.columns = ['ds', 'y']
    '''

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
        start = np.random.randint(test_time_series_duration - num_anomalies)
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
    data['y'] = data.y.apply(lambda x: 1e-1 if x < 1e-1 else x)
    data['cap'] = data.y.max()

    m = model(
        weekly_seasonality=True,
        # yearly_seasonality=True,
        holidays=sales,
        interval_width=interval_width,
        seasonality_mode='multiplicative'
    )
    with suppress_stdout_stderr():
        m.fit(data)
    forecast = m.predict(data)

    comp = pd.merge(
        forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']],
        data,
        left_on='ds',
        right_on='ds')

    comp.y = data.y.apply(lambda x: 1e-3 if x == 1e-1 else x)
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
        ax.scatter(data.ds[-test_time_series_duration:].iloc[art_anomalies],
                   data.y[-test_time_series_duration:].iloc[art_anomalies], c='g', marker='o', s=80)

        r = data.ds[-test_time_series_duration:].iloc[art_anomalies].tolist()
        for i, x in anomalies.iterrows():
            if x.ds in r:
                ax.scatter(x.ds, x.y, c='r', marker='o', s=80)
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

    return art_anomalies_fin, preds


if __name__ == '__main__':
    sales = make_sales()

    with open('configs/main_config.yaml', 'r') as f:
        config = yaml.load(f)

    for target in tqdm(config['columns']):
        ts_class = time_series_classification(target)
        for params, iw_params in zip(config['scenarios'], config['time_series_params'][ts_class]):

            m = MODELS_DICT[iw_params['model']]
            PROPHET_WIDTH = iw_params['interval_width']
            GAMMA_WIDTH = iw_params['interval_width']
            NB_WIDTH = iw_params['interval_width']

            try:
                x = validate_time_series(m, target, to_plot=True, **params)
            except:
                print(f'There was a problem with model {m}, target {target}, and params {params}')

