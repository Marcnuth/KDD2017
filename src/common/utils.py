import pandas as pd
import statsmodels.api as sm
import settings


# this function will load data from path and convert the types
def load_data(path):
    data = pd.read_csv(path)
    return convert(data)


def convert(df):

    def _to_numeric_if_exist(cols, df):
        for col in cols:
            if col in list(df.columns):
                df[col] = pd.to_numeric(df[col])

    data, cols = df, list(df.columns)

    if 'tollgate_id' in cols:
        data.tollgate_id = pd.to_numeric(data.tollgate_id)

    if 'time_window_start' in cols:
        data.time_window_start = pd.to_datetime(data.time_window_start)

    if 'time_window_end' in cols:
        data.time_window_end = pd.to_datetime(data.time_window_end)

    if 'avg_travel_time' in cols:
        data.avg_travel_time = pd.to_numeric(data.avg_travel_time)

    if 'direction' in cols:
        data.direction = pd.to_numeric(data.direction)

    if 'volume' in cols:
        data.volume = pd.to_numeric(data.volume)

    _to_numeric_if_exist([
        'link_id', 'length', 'width', 'lanes', 'lane_width'
    ], data)

    return data


def filter_same_time(df, iid, tid, d=None, t=None):
    same_iid = df.intersection_id == iid
    same_tid = df.tollgate_id == tid
    same_date = df.time_window_start.date() == d if d else True
    same_time = df.time_window_start.time() == t if t else True
    return same_iid and same_tid and same_time and same_date


def fit_arima(ts, forecast):
    tmp = sm.tsa.adfuller(ts)
    d = 0 if tmp[1] <= 0.05 else 1

    best_fit, best_pv = None, 99999
    for (p, q) in [(p, q) for p in range(4) for q in range(4)]:
        try:
            model = sm.tsa.ARIMA(ts, order=(p, d, q))
            res = model.fit(disp=-1)
            if res.pvalues.mean() < best_pv:
                best_fit = res
        except:
            continue

    assert best_fit, 'Cannot find any parameters!'
    return (best_fit.forecast(forecast)[0], best_fit.fittedvalues)


def valid_submssion(task, submit):
    if task == 'trajectories':
        sample = pd.read_csv(settings.Data.submit_trajectories)
        onkeys = ['intersection_id', 'tollgate_id', 'time_window']
        col = 'avg_travel_time'
    else:
        sample = pd.read_csv(settings.Data.submit_volume)
        onkeys = ['tollgate_id', 'direction', 'time_window']
        col = 'volume'

    errMsg = 'Shape<{0} != {1}> is not match'.format(submit.shape, sample.shape)
    assert submit.shape == sample.shape, errMsg
    merged = pd.merge(sample, submit, how='left', on=onkeys)

    assert not merged[merged[col + '_y'].isnull()].shape[0], 'Nan Found!'


def merge_time_window(data):
    df = pd.DataFrame(data)

    tmp1 = df.time_window_start.map(
        lambda x: '[' + x.strftime('%Y-%m-%d %H:%M:%S') + ','
    )
    tmp2 = df.time_window_end.map(
        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') + ')'
    )

    df['time_window'] = tmp1 + tmp2
    df.drop(['time_window_start', 'time_window_end'], axis=1, inplace=True)
    #print(df.head())
    return df


def before_holiday(dt):
    return round_holiday(dt)[0]


def after_holiday(dt):
    return round_holiday(dt)[1]


def holiday_len(dt):
    return round_holiday(dt)[2]


def start_holiday(dt):
    return round_holiday(dt)[3]


def end_holiday(dt):
    return round_holiday(dt)[4]


def round_holiday(dt):
    before = dt.weekday() == 4  # Is friday?
    after = dt.weekday() == 0  # Is monday?
    holiday_len = 0 if dt.weekday() < 5 else 2
    start = dt.weekday() == 5  # Is saturday?
    end = dt.weekday() == 6  # Is sunday?

    if dt.month == 9:
        if dt.day in range(15, 18):
            if dt.day == 15:
                start = True
            elif dt.day == 17:
                end = True

            after, before, holiday_len = False, False, 3
        elif dt.day == 14 or dt.day == 30:
            before = True
        elif dt.day == 18:
            after = True
    elif dt.month == 10:
        if dt.day in range(1, 8):
            if dt.day == 1:
                start = True
            elif dt.day == 7:
                end = True

            after, before, holiday_len = False, False, 7
        elif dt.day == 8:
            after = True

    return (before, after, holiday_len, start, end)


def get_meta(type):
    assert type in ['traj',
                    'dates', 'valids_times', 'tests_times',
                    'traj_valids_times', 'traj_tests_times',
                    'traj_valids_dates', 'traj_tests_dates',
                    'volume_valids_times', 'volume_tests_times',
                    'volume_valids_dates', 'volume_tests_dates']

    trajs = {
        'A': [2, 3],
        'B': [1, 3],
        'C': [1, 3]
    }

    # gen traj table
    ttrajs = []
    for (key, values) in sorted(trajs.items()):
        for value in values:
            ttrajs.append([key, value])

    if type == 'traj':
        return ttrajs

    dates = range(18, 25)
    tdates = []
    for d in dates:
        tdates.append([2016, 10, d])

    if type == 'dates':
        return tdates

    valids_hours = [6, 7, 15, 16]
    min_step = 20

    tvtimes = []
    for hour in valids_hours:
        for minute in range(0, 60, min_step):
            tvtimes.append([hour, minute])

    if type == 'valids_times':
        return tvtimes

    tests_hours = [8, 9, 17, 18]

    tttimes = []
    for hour in tests_hours:
        for minute in range(0, 60, min_step):
            tttimes.append([hour, minute])

    if type == 'tests_times':
        return tttimes

    t_traj_valids_times = []
    for traj in ttrajs:
        for time in tvtimes:
            t_traj_valids_times.append([*traj, *time])

    if type == 'traj_valids_times':
        return t_traj_valids_times

    t_traj_valids_dts = []
    for date in tdates:
        for traj in t_traj_valids_times:
            t_traj_valids_dts.append([*date, *traj])

    if type == 'traj_valids_dates':
        return t_traj_valids_dts

    t_traj_tests_times = []
    for traj in ttrajs:
        for time in tttimes:
            t_traj_tests_times.append([*traj, *time])

    if type == 'traj_tests_times':
        return t_traj_tests_times

    t_traj_tests_dts = []
    for date in tdates:
        for traj in t_traj_tests_times:
            t_traj_tests_dts.append([*date, *traj])

    if type == 'traj_tests_dates':
        return t_traj_tests_dts

    tollgates = [1, 2, 3]
    direction = [0, 1]
    tds = []
    for tid in tollgates:
        for direct in direction:
            # tollgate 2 only allow direction 0
            if tid == 2 and direct != 0:
                continue
            tds.append([tid, direct])

    vol_valids_times = []
    for td in tds:
        for time in tvtimes:
            vol_valids_times.append([*td, *time])

    if type == 'volume_valids_times':
        return vol_valids_times

    t_vol_valids_dts = []
    for date in tdates:
        for vol in vol_valids_times:
            t_vol_valids_dts.append([*date, *vol])

    if type == 'volume_valids_dates':
        return t_vol_valids_dts

    vol_tests_times = []
    for td in tds:
        for time in tttimes:
            vol_tests_times.append([*td, *time])

    if type == 'volume_tests_times':
        return vol_tests_times

    t_vol_tests_dts = []
    for date in tdates:
        for vol in vol_tests_times:
            t_vol_tests_dts.append([*date, *vol])

    if type == 'volume_tests_dates':
        return t_vol_tests_dts

    raise Exception('Not found the type:' + type)
