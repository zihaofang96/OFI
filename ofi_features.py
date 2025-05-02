import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Union
Symbol = str
Timestamp = pd.Timestamp

class OFICalculator:
    def __init__(
        self,
        *,
        depth: int = 10,
        window: pd.Timedelta = pd.Timedelta("1s"),   # window = h as in (t, t+h]
        pca_history: pd.Timedelta = pd.Timedelta("1D"),
    ) -> None:
        self.depth = depth
        self.window = window
        self.pca_history = pca_history

        # last LOB snapshot
        self._snap: Dict[Symbol, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        # current OFI accumulator
        self._accum: Dict[Symbol, np.ndarray] = {}
        # right edge of current window
        self._win_end: Dict[Symbol, Timestamp] = {}

        # for PCA weight estimation 
        self._pca_buf: List[Tuple[Symbol, Timestamp, np.ndarray]] = []
        self._pca_w: Dict[Symbol, np.ndarray] = {}

    # ---------------------------------------------------------------------

    def _roll_window(self, sym: Symbol, ts: Timestamp) -> None:
        """Locate time window that encloses *ts*."""
        if sym not in self._win_end:
            # first appearance → align window_end to the *ceiling* multiple of `window`
            base = pd.Timestamp.floor(ts + self.window, freq=self.window)
            self._win_end[sym] = base
            self._accum[sym] = np.zeros(self.depth)
        while ts > self._win_end[sym]:
            # step to next bucket
            self._win_end[sym] += self.window
            self._accum[sym][...] = 0.0  # reset

    # ---------------------------------------------------------------------

    def ingest_row(self, row: pd.Series) -> None:
        """Process a single post‑event LOB row."""
        ts = pd.to_datetime(row["ts_event"])
        sym: Symbol = row["symbol"]

        bid_px = np.array([row[f"bid_px_{i:02d}"] for i in range(self.depth)])
        ask_px = np.array([row[f"ask_px_{i:02d}"] for i in range(self.depth)])
        bid_sz = np.array([row[f"bid_sz_{i:02d}"] for i in range(self.depth)])
        ask_sz = np.array([row[f"ask_sz_{i:02d}"] for i in range(self.depth)])

        
        if sym in self._snap:
            l_bid_px, l_ask_px, l_bid_sz, l_ask_sz = self._snap[sym]
            omega = np.zeros(self.depth)
            for m in range(self.depth):
                # bid side
                if bid_px[m] > l_bid_px[m]:
                    ofib = bid_sz[m]
                elif bid_px[m] == l_bid_px[m]:
                    ofib = bid_sz[m] - l_bid_sz[m]
                else:
                    ofib = -l_bid_sz[m]
                # ask side
                if ask_px[m] > l_ask_px[m]:
                    ofia = -ask_sz[m]
                elif ask_px[m] == l_ask_px[m]:
                    ofia = ask_sz[m] - l_ask_sz[m]
                else:
                    ofia = l_ask_sz[m]
                omega[m] = ofib - ofia
            # accumulate into current time window
            self._roll_window(sym, ts)
            self._accum[sym] += omega
            # store omega for PCA
            self._pca_buf.append((sym, ts, omega))
        
        self._snap[sym] = (bid_px, ask_px, bid_sz, ask_sz)

    # ---------------------------------------------------------------------

    def finalize_interval(self, *, horizon_end: Timestamp) -> None:
        """Re‑estimate PCA weights on the buffer up to *horizon_end*."""
        if not self._pca_buf:
            return
        cutoff = horizon_end - self.pca_history
        # slice buffer to the last day
        recent = [(s, t, v) for (s, t, v) in self._pca_buf if t >= cutoff]
        self._pca_buf = recent  # prune old rows
        if not recent:
            return
        df = (
            pd.DataFrame(
                {"symbol": s, "ts": t, **{f"lvl_{i}": v[i] for i in range(self.depth)}}
                for s, t, v in recent
            )
            .groupby("symbol")
        )
        for sym, grp in df:
            X = grp[[f"lvl_{i}" for i in range(self.depth)]].values
            if len(X) < 10:
                continue
            # run PCA for integrated ofi
            pca = PCA(n_components=1)
            pca.fit(X)
            w = pca.components_[0]
            w /= np.sum(np.abs(w))
            self._pca_w[sym] = w

    # ---------------------------------------------------------------------
    
    def get_features(self, symbol: Symbol, timestamp: Timestamp) -> Dict[str, Union[float, np.ndarray]]:
        """Return OFI features aggregated over (t‑h, t] with *t = timestamp*."""
        if symbol not in self._accum:
            raise KeyError(f"No data for symbol {symbol}")
        # ensure `timestamp` is within the current bucket for symbol
        self._roll_window(symbol, timestamp)
        if timestamp > self._win_end[symbol]:
            raise RuntimeError("Timestamp beyond ingested data – call ingest_row first")

        # use definitions of the ofi features
        vec = self._accum[symbol].copy()
        best = float(vec[0]) 
        w = self._pca_w.get(symbol)
        integrated = float(np.dot(w, vec)) if w is not None else best
        # cross‑asset best‑level
        cross = sum(self._accum[s][0] for s in self._accum if s != symbol)
        return {
            "best_level_ofi": best,
            "multi_level_ofi": vec,
            "integrated_ofi": integrated,
            "cross_asset_ofi": float(cross),
        }

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def compute_features_for_timestamp(
    book_df: pd.DataFrame,
    ts: Union[str, Timestamp],
    *,
    symbol: str,
    depth: int = 10,
    window: str | pd.Timedelta = "1s",
) -> Dict[str, Union[float, np.ndarray]]:
    ts = pd.to_datetime(ts)
    window = pd.Timedelta(window)

    calc = OFICalculator(depth=depth, window=window)
    for _, row in book_df.sort_values("ts_event").iterrows():
        calc.ingest_row(row)
        if pd.to_datetime(row["ts_event"]) >= ts:
            break
    calc.finalize_interval(horizon_end=ts)
    return calc.get_features(symbol, ts)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, textwrap, json

    parser = argparse.ArgumentParser(
        description="Compute OFI features aggregated over (t‑h,t] for a single timestamp.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("csv", help="LOB snapshot CSV (post‑event rows)")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("timestamp", help="End‑of‑window timestamp (ISO‑8601)")
    parser.add_argument("--window", default="1s", help="Aggregation window, e.g. 1s 500ms 5min (default 1s)")
    parser.add_argument("--depth", type=int, default=10, help="LOB depth to use (default 10)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    feats = compute_features_for_timestamp(
        df,
        args.timestamp,
        symbol=args.symbol,
        depth=args.depth,
        window=args.window,
    )
    # json style output
    print(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in feats.items()}, indent=2))

"""
Sample usage in terminal:
python ofi_features.py first_25000_rows.csv AAPL 2024-10-21T11:55:00Z --window 1s
"""