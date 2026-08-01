import numpy as np
import pandas as pd

# --- Lịch khuyến mãi (giữ nguyên) ---
PROMO_SCHEDULE = [
    ('spring_sale', 3, 18, 30, 12, True),
    ('mid_year', 6, 23, 29, 18, True),
    ('fall_launch', 8, 30, 32, 10, True),
    ('year_end', 11, 18, 45, 20, True),
    ('urban_blowout', 7, 30, 33, None, 'odd'),
    ('rural_special', 1, 30, 30, 15, 'odd'),
]

# --- Ngày lễ cố định VN (THÊM MỚI) ---
VN_FIXED_HOLIDAYS = [
    (1, 1, 'new_year'), (3, 8, 'womens_day'), (4, 30, 'reunification'),
    (5, 1, 'labor_day'), (9, 2, 'national_day'), (10, 20, 'vn_womens_day'),
    (11, 11, 'dd_1111'), (12, 12, 'dd_1212'),
    (12, 24, 'christmas_eve'), (12, 25, 'christmas'),
]

TET_DATES = {
    2013: '2013-02-10', 2014: '2014-01-31', 2015: '2015-02-19',
    2016: '2016-02-08', 2017: '2017-01-28', 2018: '2018-02-16',
    2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12',
    2022: '2022-02-01', 2023: '2023-01-22', 2024: '2024-02-10',
}

def build_features(dates):
    df = pd.DataFrame({'Date': dates})
    d = df['Date']

    # 1. Calendar basics (giữ nguyên)
    df['year'] = d.dt.year
    df['month'] = d.dt.month
    df['day'] = d.dt.day
    df['dow'] = d.dt.dayofweek
    df['doy'] = d.dt.dayofyear
    df['quarter'] = d.dt.quarter
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['dim'] = d.dt.days_in_month
    df['days_to_eom'] = df['dim'] - df['day']
    df['days_from_som'] = df['day'] - 1

    # Edge of month
    for k in [1, 2, 3]:
        df[f'is_last{k}'] = (df['days_to_eom'] <= k-1).astype(int)
        df[f'is_first{k}'] = (df['days_from_som'] <= k-1).astype(int)

    # 2. Trend and regime (THÊM MỚI)
    df['t_days'] = (d - pd.Timestamp('2020-01-01')).dt.days
    df['t_years'] = df['t_days'] / 365.25
    df['regime_pre2019'] = (df['year'] <= 2018).astype(int)
    df['regime_2019'] = (df['year'] == 2019).astype(int)
    df['regime_post2019'] = (df['year'] >= 2020).astype(int)

    # 3. Fourier features (THÊM WEEKLY, MONTHLY)
    TAU = 2 * np.pi
    for k in (1, 2, 3, 4, 5):
        df[f'sin_y{k}'] = np.sin(TAU * k * df['doy'] / 365.25)
        df[f'cos_y{k}'] = np.cos(TAU * k * df['doy'] / 365.25)
    for k in (1, 2):
        df[f'sin_w{k}'] = np.sin(TAU * k * df['dow'] / 7.0)
        df[f'cos_w{k}'] = np.cos(TAU * k * df['dow'] / 7.0)
    for k in (1, 2):
        df[f'sin_m{k}'] = np.sin(TAU * k * (df['day'] - 1) / df['dim'])
        df[f'cos_m{k}'] = np.cos(TAU * k * (df['day'] - 1) / df['dim'])

    # 4. VN holidays (THÊM MỚI)
    for (m, dd_, name) in VN_FIXED_HOLIDAYS:
        df[f'hol_{name}'] = ((df['month'] == m) & (df['day'] == dd_)).astype(int)

    # 5. Black Friday (THÊM MỚI)
    def is_black_friday(dd):
        if dd.month != 11:
            return 0
        last = pd.Timestamp(year=dd.year, month=11, day=30)
        last_fri = last - pd.Timedelta(days=(last.dayofweek - 4) % 7)
        return int(dd == last_fri)
    df['hol_black_friday'] = [is_black_friday(dd) for dd in d]

    # 6. Tet features (giữ nguyên, có thể thêm tet_in_14)
    tet_lut = {y: pd.Timestamp(v) for y, v in TET_DATES.items()}
    def nearest_tet_diff(dd):
        cands = []
        if dd.year in tet_lut:
            cands.append(tet_lut[dd.year])
        if dd.year - 1 in tet_lut:
            cands.append(tet_lut[dd.year - 1])
        if dd.year + 1 in tet_lut:
            cands.append(tet_lut[dd.year + 1])
        valid = []
        for c in cands:
            diff = (dd - c).days
            if abs(diff) <= 45:
                valid.append(diff)
        if len(valid) > 0:
            return min(valid)
        return 999
    diffs = np.array([nearest_tet_diff(dd) for dd in d])
    df['tet_days_diff'] = diffs
    df['tet_in_7'] = (np.abs(diffs) <= 7).astype(int)
    df['tet_in_14'] = (np.abs(diffs) <= 14).astype(int)  # THÊM
    df['tet_before_7'] = ((diffs >= -7) & (diffs < 0)).astype(int)
    df['tet_after_7'] = ((diffs > 0) & (diffs <= 7)).astype(int)
    df['tet_on'] = (diffs == 0).astype(int)

    # 7. Promo windows (THÊM since, until, disc)
    yrs = sorted(set(df['year'].tolist()))
    for (name, sm, sd, dur, disc, recur) in PROMO_SCHEDULE:
        in_prom = np.zeros(len(df), dtype=int)
        since = np.full(len(df), -1.0)
        until = np.full(len(df), -1.0)
        discount = np.zeros(len(df))
        for y in range(min(yrs)-1, max(yrs)+2):
            if recur == 'odd' and y % 2 == 0:
                continue
            start = pd.Timestamp(year=y, month=sm, day=sd)
            end = start + pd.Timedelta(days=dur)
            mask = (d >= start) & (d <= end)
            in_prom[mask] = 1
            since[mask] = (d[mask] - start).dt.days
            until[mask] = (end - d[mask]).dt.days
            discount[mask] = disc or 0
        df[f'promo_{name}'] = in_prom
        df[f'promo_{name}_since'] = since
        df[f'promo_{name}_until'] = until
        df[f'promo_{name}_disc'] = discount

    df['is_odd_year'] = (df['year'] % 2).astype(int)
    return df