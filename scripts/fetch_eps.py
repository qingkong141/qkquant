"""一次性拉取 HS300 全量 EPS TTM 数据（约需 75 分钟，baostock 串行）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qkquant.data.fetcher import DataFetcher
from qkquant.data.storage import DuckStore

store = DuckStore()
codes = store.load_index_constituents("000300")
print(f"Fetching EPS TTM for {len(codes)} HS300 codes (est. ~75 min) ...")
fetcher = DataFetcher(store=store, source="baostock")
df = fetcher.fetch_eps_ttm(codes)
fetcher.close()
n = store.upsert_eps_ttm(df)
print(f"Done. Stored {n} EPS records for {df['code'].nunique()} codes.")
store.close()
