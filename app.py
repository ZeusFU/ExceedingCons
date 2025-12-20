import streamlit as st
import pandas as pd

st.title("Overexposure & Net PnL Analyzer (Fills + Trades)")

st.write(
    """
    Upload **both** a fills CSV and a trades CSV (like Tradovate exports), then:

    1. Choose the **Mini:Micro contract ratio** for this account (1:1, 1:5, 1:10).
    2. Enter the **max allowed simultaneous exposure in mini-equivalents**.
    3. Optionally set a **minimum overexposure duration in seconds** to ignore
       very short spikes.

    The app will:

    - Use the **root Code** (e.g. NQ/MNQ, ES/MES, MGC, SIL, MBT, MET).
    - Classify each product as **mini** or **micro** (MBT & MET treated as minis).
    - Apply mini-equivalent weights:
        - Minis (and MBT/MET): 1.0 mini-equivalent per contract.
        - Micros: (1 / chosen ratio) mini-equivalent per contract.
    - Reconstruct positions over time from fills.
    - Detect intervals where:
        - Total mini-equivalent exposure exceeds the max, and
        - The overexposure lasts at least the chosen number of seconds.
    - Cross-link those intervals to **trades** (from the trades export) by
      time overlap.
    - Use the **Net PnL** from the trades export to summarize:
        - Total Net PnL for all trades.
        - Net PnL from trades that were involved in overexposure.
    - In the trades detail:
        - Show **Bias** (LONG / SHORT).
        - Indicate whether each trade is **Mini or Micro**.
    """
)

# --- File uploaders ---------------------------------------------------------
fills_file = st.file_uploader("Upload fills CSV", type=["csv"], key="fills")
trades_file = st.file_uploader("Upload trades CSV", type=["csv"], key="trades")

# --- User-selected Mini:Micro ratio ----------------------------------------
ratio_option = st.selectbox(
    "Mini : Micro contract ratio",
    ["1:1", "1:5", "1:10"],
    index=1,  # default to 1:5
)

if ratio_option == "1:1":
    micro_weight_global = 1.0
elif ratio_option == "1:5":
    micro_weight_global = 1.0 / 5.0
elif ratio_option == "1:10":
    micro_weight_global = 1.0 / 10.0
else:
    micro_weight_global = 1.0  # fallback

max_mini_equiv = st.number_input(
    "Max allowed simultaneous exposure (in mini-equivalents)",
    min_value=0.0,
    value=5.0,
    step=0.5,
)

min_duration_seconds = st.number_input(
    "Minimum overexposure duration (seconds)",
    min_value=0,
    value=0,
    step=1,
)

# --------------------------------------------------------------------
# Root Code classification using instrument list logic
# --------------------------------------------------------------------
MINI_ROOTS = [
    # CME FX & index minis
    "6A", "6B", "6C", "6E", "6J", "6N", "6S",
    "NQ", "RTY", "ES", "NKD",
    "HE", "LE",
    # Metals (full/mini)
    "HG", "GC", "PL", "SI",
    # Grains & equity
    "ZC", "ZS", "ZM", "ZL", "ZW", "YM",
    # Energies
    "CL", "QM", "HO", "NG", "RB", "QG",
    # Cryptos treated as minis:
    "MBT", "MET",
]

MICRO_ROOTS = [
    # FX & index micros
    "M6A", "M6E",
    "MNQ", "M2K", "MES",
    "MYM",
    # Metals micros
    "MGC", "SIL",
    # Energy micros
    "MCL",
]

ROOT_TYPES = {root: "mini" for root in MINI_ROOTS}
ROOT_TYPES.update({root: "micro" for root in MICRO_ROOTS})

# Known roots list for prefix matching (longest-first)
KNOWN_ROOTS = sorted(ROOT_TYPES.keys(), key=len, reverse=True)


def get_root_code(asset: str) -> str:
    """
    Given a symbol (e.g. 'MNQZ5', 'ESM4'),
    return the ROOT Code (e.g. 'MNQ', 'ES') by prefix match.
    """
    code = str(asset).strip().upper()
    for root in KNOWN_ROOTS:
        if code.startswith(root):
            return root
    return code  # unknown root, treated as mini by default


def get_instrument_weight(asset: str, micro_weight: float) -> float:
    """
    Return the mini-equivalent weight for a given symbol based on its
    root Code and the user-selected mini:micro ratio.

    - Minis: 1.0
    - Micros: micro_weight (1/ratio)
    - Unknown roots: default to 1.0
    """
    root = get_root_code(asset)
    type_ = ROOT_TYPES.get(root, "mini")
    if type_ == "mini":
        return 1.0
    elif type_ == "micro":
        return micro_weight
    else:
        return 1.0


def find_overexposure_with_details(
    df: pd.DataFrame,
    max_mini_equiv: float,
    micro_weight: float,
    min_duration_seconds: int,
):
    """
    From a fills DataFrame with columns:
        - action (Buy/Sell)
        - asset  (symbol, e.g. MNQZ5, NQZ5, SILH4, etc.)
        - quantity
        - timestamp

    Returns:
        intervals_summary: list of dicts (one row per overexposed interval)
        detail_rows: list of dicts (one row per symbol per overexposed interval)
        cleaned_df: cleaned & time-ordered fills used in the computation
    """
    required_cols = {"action", "asset", "quantity", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fills CSV is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["quantity", "timestamp_dt"])
    df = df.sort_values("timestamp_dt").reset_index(drop=True)

    if df.empty:
        return [], [], df

    positions = {}  # asset (full symbol) -> net position (signed)
    intervals_summary = []
    detail_rows = []
    group_id = 0

    for i, row in df.iterrows():
        # Interval between fill i-1 and i uses positions BEFORE applying fill i
        if i > 0:
            start = df.loc[i - 1, "timestamp_dt"]
            end = row["timestamp_dt"]
            duration = (end - start).total_seconds()

            # Snapshot mini-equivalent exposure per symbol at this interval
            snapshot = {}  # asset -> info dict
            for asset, pos in positions.items():
                if pos == 0:
                    continue
                contracts = abs(pos)
                if contracts == 0:
                    continue

                root = get_root_code(asset)
                weight = get_instrument_weight(asset, micro_weight)
                exposure = contracts * weight  # mini-equivalent exposure

                snapshot[asset] = {
                    "root": root,
                    "contracts": contracts,
                    "weight": weight,
                    "exposure": exposure,
                }

            if snapshot:
                total_exposure = sum(info["exposure"] for info in snapshot.values())

                # Duration filter: skip if interval is shorter than requested
                if duration >= min_duration_seconds:
                    # Overexposure condition: total exposure > max
                    if total_exposure > max_mini_equiv:
                        group_id += 1

                        # Aggregate exposure by root for the summary
                        root_exposure = {}
                        for info in snapshot.values():
                            r = info["root"]
                            root_exposure[r] = root_exposure.get(r, 0.0) + info["exposure"]

                        intervals_summary.append(
                            {
                                "Group ID": group_id,
                                "Interval Start": start,
                                "Interval End": end,
                                "Duration (sec)": duration,
                                "Total Mini-Equiv Exposure": total_exposure,
                                "Exposure by Root (mini-equiv)": ", ".join(
                                    f"{r}: {root_exposure[r]:.2f}"
                                    for r in sorted(root_exposure)
                                ),
                            }
                        )

                        # Detailed rows (one per symbol per interval)
                        for asset, info in snapshot.items():
                            detail_rows.append(
                                {
                                    "Group ID": group_id,
                                    "Interval Start": start,
                                    "Interval End": end,
                                    "Duration (sec)": duration,
                                    "Symbol": asset,
                                    "Root Code": info["root"],
                                    "Position Contracts": info["contracts"],
                                    "Mini-Equiv Weight": info["weight"],
                                    "Mini-Equiv Exposure": info["exposure"],
                                }
                            )

        # Apply current fill to update positions
        qty = row["quantity"]
        action = str(row["action"]).strip().lower()
        asset = row["asset"]

        if action == "buy":
            delta = qty
        elif action == "sell":
            delta = -qty
        else:
            # Unknown action, ignore
            continue

        positions[asset] = positions.get(asset, 0.0) + delta

    return intervals_summary, detail_rows, df


def parse_money_to_float(series: pd.Series) -> pd.Series:
    """
    Convert a Series of money strings like '$190.50' to floats 190.50.
    """
    return pd.to_numeric(series.astype(str).str.replace("[$,]", "", regex=True), errors="coerce")


def infer_bias_column(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort inference of LONG/SHORT bias per trade.
    Looks for columns like 'Side', 'Direction', or 'Bias'.
    """
    bias_source_col = None
    for col in df.columns:
        lc = col.lower()
        if "side" in lc or "direction" in lc or "bias" in lc:
            bias_source_col = col
            break

    def map_bias(val):
        v = str(val).strip().lower()
        if v in ["long", "buy", "b"]:
            return "LONG"
        if v in ["short", "sell", "s"]:
            return "SHORT"
        return "UNKNOWN"

    if bias_source_col is not None:
        return df[bias_source_col].map(map_bias)
    else:
        # Fallback: unknown if we can't find any directional column
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)


if fills_file is not None and trades_file is not None:
    try:
        fills_df = pd.read_csv(fills_file)
        trades_df = pd.read_csv(trades_file)

        st.subheader("Raw fills (first 50 rows)")
        st.dataframe(fills_df.head(50))

        st.subheader("Raw trades (first 50 rows)")
        st.dataframe(trades_df.head(50))

        # --- Overexposure detection from fills --------------------------------
        intervals_summary, detail_rows, cleaned_fills = find_overexposure_with_details(
            fills_df,
            max_mini_equiv=max_mini_equiv,
            micro_weight=micro_weight_global,
            min_duration_seconds=min_duration_seconds,
        )

        st.subheader("Cleaned & time-ordered fills used for calculation")
        fills_cols_to_show = [
            c for c in ["timestamp_dt", "timestamp", "action", "asset", "quantity", "price"]
            if c in cleaned_fills.columns
        ]
        st.dataframe(
            cleaned_fills[fills_cols_to_show].head(200),
            use_container_width=True,
        )

        # --- Trades preprocessing ---------------------------------------------
        trades_df = trades_df.copy()
        trades_df["Open_dt"] = pd.to_datetime(trades_df["Open Time"], errors="coerce")
        trades_df["Close_dt"] = pd.to_datetime(trades_df["Close Time"], errors="coerce")
        trades_df["Duration_sec"] = (trades_df["Close_dt"] - trades_df["Open_dt"]).dt.total_seconds()

        # Net PnL
        if "Net Profit" in trades_df.columns:
            trades_df["NetProfit_val"] = parse_money_to_float(trades_df["Net Profit"])
        else:
            trades_df["NetProfit_val"] = pd.NA

        # Root & mini/micro classification per trade
        trades_df["Root Code"] = trades_df["Symbol"].apply(get_root_code)
        trades_df["Size Type"] = trades_df["Root Code"].apply(
            lambda r: "Micro" if ROOT_TYPES.get(r, "mini") == "micro" else "Mini"
        )

        # Bias (LONG / SHORT / UNKNOWN)
        trades_df["Bias"] = infer_bias_column(trades_df)

        # --- Instrument mapping summary (from fills) --------------------------
        st.subheader("Instrument root codes & mini-equivalent weights (under selected ratio)")
        unique_assets = sorted(cleaned_fills["asset"].astype(str).str.upper().unique())
        mapping_rows = []
        seen_roots = set()
        for sym in unique_assets:
            root = get_root_code(sym)
            weight = get_instrument_weight(sym, micro_weight_global)
            if root not in seen_roots:
                seen_roots.add(root)
                mapping_rows.append(
                    {
                        "Root Code": root,
                        "Example Symbol (from fills)": sym,
                        "Type": ROOT_TYPES.get(root, "mini (default)"),
                        "Mini-Equiv Weight (per 1 contract)": weight,
                    }
                )
        if mapping_rows:
            mapping_df = pd.DataFrame(mapping_rows).sort_values("Root Code")
            st.dataframe(mapping_df, use_container_width=True)

        # --- Overexposed intervals -------------------------------------------
        st.subheader("Overexposed intervals summary")

        if not intervals_summary:
            if min_duration_seconds > 0:
                st.info(
                    f"No intervals found where total exposure exceeded {max_mini_equiv} "
                    f"mini-equivalents for at least {min_duration_seconds} seconds "
                    f"under a {ratio_option} Mini:Micro ratio."
                )
            else:
                st.info(
                    f"No intervals found where total exposure exceeded {max_mini_equiv} "
                    f"mini-equivalents under a {ratio_option} Mini:Micro ratio."
                )
            overexposed_trade_indices = set()
        else:
            summary_df = pd.DataFrame(intervals_summary)
            st.dataframe(summary_df, use_container_width=True)

            st.success(
                f"Found {len(summary_df)} interval(s) where total exposure exceeded "
                f"{max_mini_equiv} mini-equivalents under a {ratio_option} Mini:Micro ratio"
                + (f" and lasted at least {min_duration_seconds} seconds." if min_duration_seconds > 0 else ".")
            )

            detail_df = pd.DataFrame(detail_rows)

            # Color rows by Group ID so overexposed intervals are visually linked
            color_cycle = [
                "#ffcccc",
                "#ccffcc",
                "#ccccff",
                "#fff2cc",
                "#f4cccc",
                "#d9d2e9",
                "#cfe2f3",
                "#d0e0e3",
            ]
            unique_groups = sorted(detail_df["Group ID"].unique())
            group_to_color = {
                gid: color_cycle[idx % len(color_cycle)]
                for idx, gid in enumerate(unique_groups)
            }
            detail_df["Color"] = detail_df["Group ID"].map(group_to_color)

            def color_rows(row):
                color = row["Color"]
                return [
                    f"background-color: {color}" if col != "Color" else ""
                    for col in row.index
                ]

            styled_detail = detail_df.style.apply(color_rows, axis=1)

            st.subheader("Detailed per-symbol overexposure (color-coded)")
            st.dataframe(styled_detail, use_container_width=True)

            st.caption(
                "Each color represents an interval where total mini-equivalent exposure "
                "was above your limit. Rows with the same color belong to the same "
                "overexposure interval, showing each symbol's reconstructed position "
                f"under the selected {ratio_option} Mini:Micro ratio."
            )

            # --- Cross-link overexposed intervals to trades -------------------
            overexposed_trade_indices = set()
            intervals_df = pd.DataFrame(intervals_summary)

            st.subheader("Overexposed intervals with matching trades")

            for gid in sorted(intervals_df["Group ID"].unique()):
                inter_row = intervals_df[intervals_df["Group ID"] == gid].iloc[0]
                start = inter_row["Interval Start"]
                end = inter_row["Interval End"]

                # Trades whose lifespan overlaps this interval
                mask = (trades_df["Open_dt"] < end) & (trades_df["Close_dt"] > start)
                trades_for_interval = trades_df[mask].copy()
                overexposed_trade_indices.update(trades_for_interval.index.tolist())

                with st.expander(
                    f"Interval {gid}: {start} → {end} "
                    f"(Exposure: {inter_row['Total Mini-Equiv Exposure']:.2f})"
                ):
                    st.markdown("**Exposure snapshot (per symbol)**")
                    exp_rows = detail_df[detail_df["Group ID"] == gid].copy()
                    exp_cols = [
                        "Symbol",
                        "Root Code",
                        "Position Contracts",
                        "Mini-Equiv Weight",
                        "Mini-Equiv Exposure",
                        "Duration (sec)",
                    ]
                    st.dataframe(exp_rows[exp_cols], use_container_width=True)

                    st.markdown("**Trades overlapping this interval**")
                    if trades_for_interval.empty:
                        st.info("No trades from the trades export overlap this interval.")
                    else:
                        # Prepare trade detail: Bias, Size Type, Net PnL, etc.
                        trade_cols = [
                            "Symbol",
                            "Root Code",
                            "Size Type",
                            "Bias",
                            "Volume",
                            "Open Time",
                            "Close Time",
                            "Duration_sec",
                            "Net Profit",
                            "NetProfit_val",
                            "Open_dt",  # keep for sorting
                        ]
                        existing_trade_cols = [c for c in trade_cols if c in trades_for_interval.columns]

                        display_df = trades_for_interval.sort_values("Open_dt")
                        st.dataframe(
                            display_df[existing_trade_cols],
                            use_container_width=True,
                        )
                        st.caption(
                            "Each trade shows its **Bias** (LONG/SHORT) and whether it is "
                            "**Mini or Micro** (Size Type), based on the root Code."
                        )

        # --- Net PnL summary from trades -------------------------------------
        st.subheader("Net PnL Summary (from trades export)")

        if "NetProfit_val" in trades_df.columns and trades_df["NetProfit_val"].notna().any():
            total_net_pnl = trades_df["NetProfit_val"].sum()

            pnl_overexposed_trades = trades_df.loc[
                list(overexposed_trade_indices), "NetProfit_val"
            ].sum() if overexposed_trade_indices else 0.0

            col1, col2 = st.columns(2)
            col1.metric(
                "Total Net PnL (all trades)",
                f"{total_net_pnl:,.2f}",
                help="Sum of Net Profit from the trades export.",
            )
            col2.metric(
                "Net PnL from overexposed trades",
                f"{pnl_overexposed_trades:,.2f}",
                help=(
                    "Sum of Net Profit from trades that overlapped at least one "
                    "overexposure interval under the current settings."
                ),
            )

            # Detailed Net PnL per trade, flagging overexposed ones
            st.subheader("Trades detail with Net PnL and overexposure flag")

            trades_df["Involved in Overexposure"] = trades_df.index.isin(
                overexposed_trade_indices
            )
            trades_detail_cols = [
                "Symbol",
                "Root Code",
                "Size Type",
                "Bias",
                "Volume",
                "Open Time",
                "Close Time",
                "Duration_sec",
                "Net Profit",
                "NetProfit_val",
                "Involved in Overexposure",
                "Open_dt",
            ]
            existing_cols = [c for c in trades_detail_cols if c in trades_df.columns]
            trades_detail_df = trades_df[existing_cols].copy()

            trades_detail_df = trades_detail_df.sort_values(
                ["Involved in Overexposure", "Open_dt"],
                ascending=[False, True],
            )

            st.dataframe(
                trades_detail_df,
                use_container_width=True,
            )

            st.caption(
                "Net PnL is taken directly from the trades export (**Net Profit**). "
                "Trades flagged as 'Involved in Overexposure' are those whose "
                "lifespan overlapped at least one overexposed interval. "
                "Each trade shows its **Bias** and whether it is **Mini or Micro**."
            )
        else:
            st.warning(
                "No valid 'Net Profit' column found in the trades CSV, so Net PnL "
                "could not be computed. Make sure your trades export includes "
                "a 'Net Profit' column."
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("Upload **both** a fills CSV and a trades CSV to begin.")
