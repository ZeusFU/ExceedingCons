import streamlit as st
import pandas as pd

st.title("Overlapping Exposure Checker (Mini-Equivalent, Root Code Aware)")

st.write(
    """
    Upload a **fills CSV** (like `fills_export.csv`), then enter the
    **max allowed simultaneous exposure in mini-equivalents**.

    The app:

    1. Uses the **root Code** (as per the My Funded Futures instrument list)
       from each symbol in the CSV.
    2. Assigns a **mini-equivalent weight** per contract based on whether the
       product is a mini or a micro (e.g. MNQ = 0.1 mini, M6E = 0.2 mini).
    3. Treats **MBT** and **MET** as **1 full mini** each, even though they are
       named "Micro".
    4. Reconstructs positions over time from fills.
    5. Finds time intervals where:
       - At least **two different root Codes** have open positions, and
       - The **total mini-equivalent exposure** exceeds the max.
    6. Shows:
       - A summary of those intervals.
       - A **color-coded detailed table** with both reconstructed trades
         in each interval.
    """
)

uploaded_file = st.file_uploader("Upload fills CSV", type=["csv"])

max_mini_equiv = st.number_input(
    "Max allowed simultaneous exposure (in mini-equivalents)",
    min_value=0.0,
    value=5.0,
    step=0.5,
)

# --------------------------------------------------------------------
# Instrument metadata using ROOT Code (from MFF instrument list)
# --------------------------------------------------------------------
# Weight here is the "mini-equivalent" exposure per 1 contract of that root.
# - Minis and "full-size" contracts: weight = 1.0
# - Micros: weight < 1.0 based on true mini:micro ratio
# - MBT and MET are treated as 1.0 mini each (per MFF rule and your note)
INSTRUMENT_WEIGHTS = {
    # CME FX & index minis
    "6A": 1.0,
    "6B": 1.0,
    "6C": 1.0,
    "6E": 1.0,
    "6J": 1.0,
    "6N": 1.0,
    "6S": 1.0,
    "NQ": 1.0,
    "RTY": 1.0,
    "ES": 1.0,
    "NKD": 1.0,
    "HE": 1.0,
    "LE": 1.0,

    # Micros (true micro sizes)
    # 6A vs M6A: tick 5.00 vs 1.00  -> ratio 5:1 -> micro = 0.2 mini
    "M6A": 0.2,
    # 6E vs M6E: tick 6.25 vs 1.25 -> ratio 5:1 -> micro = 0.2 mini
    "M6E": 0.2,
    # NQ vs MNQ: tick 5 vs 0.50    -> ratio 10:1 -> micro = 0.1 mini
    "MNQ": 0.1,
    # RTY vs M2K: tick 5 vs 0.50   -> ratio 10:1 -> micro = 0.1 mini
    "M2K": 0.1,
    # ES vs MES: tick 12.5 vs 1.25 -> ratio 10:1 -> micro = 0.1 mini
    "MES": 0.1,

    # Cryptos (MBT/MET treated as minis)
    "MBT": 1.0,  # Micro Bitcoin but treated as full mini for risk
    "MET": 1.0,  # Micro Ethereum but treated as full mini per your rule

    # COMEX metals
    "HG": 1.0,
    "GC": 1.0,
    "PL": 1.0,
    "SI": 1.0,
    # Micro Gold: 10.0 vs 1.0 -> 0.1
    "MGC": 0.1,
    # Micro Silver: 25.0 vs 5.0 -> 0.2
    "SIL": 0.2,

    # CBOT grains & equity
    "ZC": 1.0,
    "ZS": 1.0,
    "ZM": 1.0,
    "ZL": 1.0,
    "ZW": 1.0,
    "YM": 1.0,
    # Micro E-mini Dow: 5 vs 0.5 -> 0.1
    "MYM": 0.1,

    # NYMEX energies
    "CL": 1.0,
    "QM": 1.0,
    "HO": 1.0,
    "NG": 1.0,
    "RB": 1.0,
    "QG": 1.0,
    # Micro Crude Oil: 10 vs 1 -> 0.1
    "MCL": 0.1,
}

# Sort known roots longest-first so prefix matching prefers longer codes
KNOWN_ROOTS = sorted(INSTRUMENT_WEIGHTS.keys(), key=len, reverse=True)


def get_root_code(asset: str) -> str:
    """
    Given a symbol from the fills CSV (e.g. 'MNQZ5', 'ESM4'),
    return the ROOT Code as defined by the instrument list (e.g. 'MNQ', 'ES').

    Strategy:
    - Uppercase the asset string.
    - Find the longest instrument Code from INSTRUMENT_WEIGHTS that is a prefix.
    - If none match, fall back to treating the entire symbol as the root.
    """
    code = str(asset).strip().upper()
    for root in KNOWN_ROOTS:
        if code.startswith(root):
            return root
    return code  # fallback: unknown root, will default to weight 1.0


def get_instrument_weight(asset: str) -> float:
    """
    Return the mini-equivalent weight for a given symbol based on its root Code.
    Unknown roots default to 1.0 mini-equivalent per contract.
    """
    root = get_root_code(asset)
    return INSTRUMENT_WEIGHTS.get(root, 1.0)


def find_overlapping_exposure_with_details(df: pd.DataFrame, max_mini_equiv: float):
    """
    From a fills DataFrame with columns:
        - action (Buy/Sell)
        - asset  (symbol, e.g. MNQZ5, NQZ5, SILH4, etc.)
        - quantity
        - timestamp

    Returns:
        intervals_summary: list of dicts (one row per violating interval)
        detail_rows: list of dicts (one row per symbol per violating interval)
        cleaned_df: cleaned & time-ordered fills used in the computation
    """
    required_cols = {"action", "asset", "quantity", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

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

            # Snapshot mini-equivalent exposure per symbol at this interval
            snapshot = {}  # asset -> info dict
            for asset, pos in positions.items():
                if pos == 0:
                    continue
                contracts = abs(pos)
                if contracts == 0:
                    continue

                root = get_root_code(asset)
                weight = get_instrument_weight(asset)
                exposure = contracts * weight  # mini-equivalent exposure

                snapshot[asset] = {
                    "root": root,
                    "contracts": contracts,
                    "weight": weight,
                    "exposure": exposure,
                }

            if snapshot:
                total_exposure = sum(info["exposure"] for info in snapshot.values())
                active_roots = {info["root"] for info in snapshot.values()}

                # Condition:
                #   - at least two different root Codes live (different instruments)
                #   - total exposure > max_mini_equiv
                if len(active_roots) >= 2 and total_exposure > max_mini_equiv:
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
                            "Total Mini-Equiv Exposure": total_exposure,
                            "Active Roots (mini-equiv)": ", ".join(
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


if uploaded_file is not None:
    try:
        fills_df = pd.read_csv(uploaded_file)

        st.subheader("Raw fills (first 50 rows)")
        st.dataframe(fills_df.head(50))

        intervals_summary, detail_rows, cleaned_df = find_overlapping_exposure_with_details(
            fills_df, max_mini_equiv
        )

        st.subheader("Cleaned & time-ordered fills used for calculation")
        st.dataframe(
            cleaned_df[["timestamp_dt", "action", "asset", "quantity"]].head(200),
            use_container_width=True,
        )

        # Show mapping from symbols in the file to root codes & weights
        st.subheader("Instrument root codes & mini-equivalent weights used")
        unique_assets = sorted(cleaned_df["asset"].astype(str).str.upper().unique())
        mapping_rows = []
        seen_roots = set()
        for sym in unique_assets:
            root = get_root_code(sym)
            weight = get_instrument_weight(sym)
            if root not in seen_roots:
                seen_roots.add(root)
                mapping_rows.append(
                    {
                        "Root Code": root,
                        "Example Symbol": sym,
                        "Mini-Equiv Weight (per 1 contract)": weight,
                    }
                )
        if mapping_rows:
            mapping_df = pd.DataFrame(mapping_rows).sort_values("Root Code")
            st.dataframe(mapping_df, use_container_width=True)

        st.subheader("Overlapping intervals summary")

        if not intervals_summary:
            st.info(
                f"No time intervals found where overlapping positions in different "
                f"root Codes exceeded {max_mini_equiv} mini-equivalents."
            )
        else:
            summary_df = pd.DataFrame(intervals_summary)
            st.dataframe(summary_df, use_container_width=True)

            st.success(
                f"Found {len(summary_df)} interval(s) where overlapping positions "
                f"in different instruments exceeded {max_mini_equiv} mini-equivalents."
            )

            st.subheader("Detailed per-symbol overlapping exposure (color-coded)")

            detail_df = pd.DataFrame(detail_rows)

            # Assign a color to each Group ID so rows from the same overlapping
            # interval share the same background.
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

            st.dataframe(styled_detail, use_container_width=True)

            st.caption(
                "Each color represents one interval where total mini-equivalent "
                "exposure across multiple instruments exceeded the max. "
                "Rows with the same color belong to the same overlapping interval, "
                "showing both reconstructed trades independently but grouped by root Code."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a fills CSV to begin.")
