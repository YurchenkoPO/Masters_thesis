import pandas as pd
import warnings
warnings.filterwarnings("ignore")

START_DATE = '2021-02-01'


df = pd.read_parquet('data/raw/df.parquet')
df = df[df.product_creation_date >= START_DATE]

stats = pd.read_parquet('data/raw/stats.parquet')
stats.loc[:, 'product_creation_date'] = pd.to_datetime(stats.product_creation_date)
stats = stats[stats.product_creation_date >= START_DATE]
segments_sold_per_day = stats.groupby('product_creation_date')['sum_seg_count'].agg('sum')
all_products = df.product_type.value_counts()

df = df[df.product_creation_date < df.product_creation_date.max()]

df['departure'] = df['departure'].fillna('XXX')
df['arrival'] = df['arrival'].fillna('XXX')
df['pos'] = df['pos'].fillna('Undefined')
df['ond'] = df.departure+'-' + df.arrival
df = df.groupby(['product_type', 'product_creation_date', 'ond'])[['sum_price', 'sum_quantity', 'count_rows']].agg('sum').reset_index()

tracked_pairs = df.groupby(['product_type', 'ond'])['sum_quantity'].agg('sum')
tracked_pairs = tracked_pairs[tracked_pairs > 1000].reset_index()

tracked_products = set(tracked_pairs.product_type)
tracked_onds = set(tracked_pairs.ond)

df.loc[~df.product_type.isin(tracked_products), 'product_type'] = 'other'
df.loc[~df.ond.isin(tracked_onds), 'ond'] = 'XXX-XXX'
df = df.groupby(['product_type', 'product_creation_date', 'ond'])[['sum_price', 'sum_quantity', 'count_rows']].agg('sum').reset_index()

untracked_ids = df.join(tracked_pairs.set_index(['product_type', 'ond']).rename(columns={'sum_quantity': 'is_tracked'}), on=['product_type', 'ond'])['is_tracked'].isna()

df.loc[untracked_ids, 'ond'] = 'XXX-XXX'
df = df.groupby(['product_type', 'product_creation_date', 'ond'])[['sum_price', 'sum_quantity', 'count_rows']].agg('sum').reset_index()

d1 = df.groupby(['product_type', 'product_creation_date'])['sum_quantity'].agg('sum').reset_index()

onds = set(df.ond)

tbl_detailed = df.groupby(['product_creation_date', 'product_type', 'ond'])['sum_quantity'].agg(sum).unstack().unstack().fillna(0)
tbl_detailed.columns = [tuple(c) for c in tbl_detailed.columns]
pairs_to_drop = tbl_detailed.sum(axis=0)
pairs_to_drop =pairs_to_drop[pairs_to_drop == 0].index
tbl_detailed = tbl_detailed.drop(columns=pairs_to_drop)

# Объединяем всё в итоговую таблицу tbl
tbl = d1.set_index(['product_creation_date', 'product_type']).unstack().fillna(0)
cols = [c[1] for c in tbl.columns]
tbl.columns = cols
tbl_total = tbl.loc[:, cols].sum(axis=1).rename('total')
tbl = pd.concat([tbl_total, segments_sold_per_day, tbl, tbl_detailed], axis=1)

tbl = tbl[:-7]
tbl.columns = [str(col) for col in tbl.columns]
tbl.to_parquet('data/processed/processed.parquet')
