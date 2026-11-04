"""錨（anchor）の純粋関数。

シグナル（買い +1 / 売り -1）を、先の値動きと突き合わせて採点する固定の物差し。
DB や他 BC には触れず、DataFrame だけを受け取って返す。

- 価格は adj_close があればそれ、無ければ close を使う
- 騰落率 0 はシグナル側では外れ、ベースライン側では up/down のどちらにも数えない
- ベースラインはシグナルの日ごと: その日に足のある全銘柄のうち、
  シグナルと同じ向きに動いた銘柄の割合
"""

import pandas as pd

HORIZONS: tuple[int, ...] = (1, 5, 20)
ANCHOR_HORIZON = 5
# 1日の騰落率の絶対値がこれ以上なら、分割・併合などの価格の飛びとみなして採点から外す
SPLIT_JUMP = 0.45


def price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """[symbol, ts, close, adj_close] を [symbol, date, px] にする。"""
    px = raw["adj_close"].where(raw["adj_close"].notna(), raw["close"])
    out = pd.DataFrame(
        {
            "symbol": raw["symbol"].to_numpy(),
            "ts": pd.to_datetime(raw["ts"]).to_numpy(),
            "px": px.to_numpy(),
        }
    )
    # 価格の欠損行は捨てる（残すと行の並びの h 本先が欠損をまたぎ、飛びを見逃す）
    out = out.dropna(subset=["px"]).sort_values(["symbol", "ts"])
    out["date"] = pd.to_datetime(out["ts"]).dt.date
    # 同じ日に ts が複数あれば最後の足だけ残す（行の並びで h 本先を取るため日付は一意にする）
    out = out.drop_duplicates(["symbol", "date"], keep="last")
    return out[["symbol", "date", "px"]].reset_index(drop=True)


def forward_returns(px: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """銘柄ごとの行の並びで h 本先の騰落率 ret_h と、窓内の飛び split_h を付ける。"""
    out = px.dropna(subset=["px"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    grp = out.groupby("symbol", sort=False)["px"]
    jump = (grp.pct_change().abs() >= SPLIT_JUMP).astype(bool)
    jump_grp = jump.groupby(out["symbol"], sort=False)
    for h in horizons:
        out[f"ret_{h}"] = grp.shift(-h) / out["px"] - 1
        split = pd.Series(False, index=out.index)
        for k in range(1, h + 1):
            split |= jump_grp.shift(-k).fillna(False).astype(bool)
        out[f"split_{h}"] = split
    return out


def daily_baseline(fwd: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """日ごとの上昇銘柄の割合 up_rate と下落銘柄の割合 down_rate。

    ret が NaN の行は除く。0 はどちらにも入らない。
    """
    ret = fwd.loc[fwd[f"ret_{horizon}"].notna(), ["date", f"ret_{horizon}"]]
    r = ret[f"ret_{horizon}"]
    flags = pd.DataFrame(
        {
            "date": ret["date"],
            "up_rate": (r > 0).astype(float),
            "down_rate": (r < 0).astype(float),
        }
    )
    return flags.groupby("date", as_index=False).mean()


def score_signals(
    signals: pd.DataFrame,
    fwd: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[pd.DataFrame, int]:
    """シグナルを採点する。

    Returns:
        ([date, symbol, key, direction, horizon, hit, baseline, signed_return],
         価格の飛びで除外した (シグナル, horizon) の件数)
    """
    cols = ["date", "symbol", "key", "direction", "horizon", "hit", "baseline", "signed_return"]
    sig = signals.loc[signals["direction"] != 0, ["date", "symbol", "key", "direction"]]
    frames: list[pd.DataFrame] = []
    n_excluded = 0
    for h in horizons:
        base = daily_baseline(fwd, h)
        m = sig.merge(
            fwd[["date", "symbol", f"ret_{h}", f"split_{h}"]],
            on=["date", "symbol"],
            how="inner",
        )
        m = m[m[f"ret_{h}"].notna()]
        n_excluded += int(m[f"split_{h}"].sum())
        m = m[~m[f"split_{h}"]].merge(base, on="date", how="left")
        ret = m[f"ret_{h}"]
        signed = m["direction"] * ret
        frames.append(
            pd.DataFrame(
                {
                    "date": m["date"],
                    "symbol": m["symbol"],
                    "key": m["key"],
                    "direction": m["direction"],
                    "horizon": h,
                    "hit": signed > 0,
                    "baseline": m["up_rate"].where(m["direction"] > 0, m["down_rate"]),
                    "signed_return": signed,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=cols), n_excluded
    return pd.concat(frames, ignore_index=True)[cols], n_excluded


def summarize(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """by ごとに件数・的中率・ベースライン率・edge・平均の符号付き騰落率を出す。"""
    g = scored.groupby(by, as_index=False).agg(
        n=("hit", "size"),
        hit_rate=("hit", "mean"),
        baseline_rate=("baseline", "mean"),
        mean_signed_return=("signed_return", "mean"),
    )
    g["edge"] = g["hit_rate"] - g["baseline_rate"]
    return g[by + ["n", "hit_rate", "baseline_rate", "edge", "mean_signed_return"]]
