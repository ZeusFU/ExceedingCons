import streamlit as st
import pandas as pd

st.title("Overlapping Exposure Checker (Mini-Equivalent, Root Code Aware)")

st.write(
    """
    Upload a **fills CSV** (like `fills_export.csv`), then:

    1. Choose the **Mini:Micro contract ratio** for this account (1:1, 1:5, 1:10).
    2. Enter the **max allowed simultaneous exposure in mini-equivalents**.

    The app:

    - Uses the **root Code** as per the My Funded Futures instrument list.
    - Classifies each product as **mini** or **micro**.
    - Applies mini-equivalent weights:
        - Minis: 1.0 mini-equivalent per contract.
        - Micros: (1 / chosen ratio) mini-equivalent per contract.
        - **MBT** and **MET** are treated as minis for risk (weight = 1.0).
    - Reconstructs positions over time from fills.
    - Finds time intervals where:
        - At least **two different root Codes** have open positions, and
        - The **total mini-equivalent exposure** exceeds the max.
    - Shows a summary and a **color-coded detailed table** of both reconstructed trades.
    """
)

uploaded_file = st.file_uploader("Upload fills CSV", type=["csv"])

# --- User-selected Mini:Micro ratio -----------------------------------------
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

# --------------------------------------------------------------------
# Root Code classification using instrument list logic
# --------------------------------------------------------------------
# We'll classify each root as:
#   "mini"  -> 1.0 mini-equivalent per contract
#   "micro" -> micro_weight_global mini-equivalent per contract
#
# MBT and MET are treated as "mini" for risk purposes, even though
# they are named Micro products.

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
    Given a symbol from the fills CSV (e.g. 'MNQZ5', 'ESM4'),
    return the ROOT Code as per the instrument list (e.g. 'MNQ', 'ES').

    Strategy:
    - Uppercase the asset string.
    - Find the longest known root that is a prefix.
    - If none match, fall back to treating the entire symbol as the root.
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


def find_overlapping_exposure_with_details(
    df: pd.DataFrame,
    max_mini_equiv: float,
    micro_weight: float,
):
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
            fills_df,
            max_mini_equiv=max_mini_equiv,
            micro_weight=micro_weight_global,
        )

        st.subheader("Cleaned & time-ordered fills used for calculation")
        st.dataframe(
            cleaned_df[["timestamp_dt", "action", "asset", "quantity"]].head(200),
            use_container_width=True,
        )

        # Show mapping from root codes & symbols to weights under the selected ratio
        st.subheader("Instrument root codes & mini-equivalent weights (under selected ratio)")
        unique_assets = sorted(cleaned_df["asset"].astype(str).str.upper().unique())
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
                        "Example Symbol": sym,
                        "Type": ROOT_TYPES.get(root, "mini (default)"),
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
                f"root Codes exceeded {max_mini_equiv} mini-equivalents under a "
                f"{ratio_option} Mini:Micro ratio."
            )
        else:
            summary_df = pd.DataFrame(intervals_summary)
            st.dataframe(summary_df, use_container_width=True)

            st.success(
                f"Found {len(summary_df)} interval(s) where overlapping positions "
                f"in different instruments exceeded {max_mini_equiv} mini-equivalents "
                f"under a {ratio_option} Mini:Micro ratio."
            )

            st.subheader("Detailed per-symbol overlapping exposure (color-coded)")

            detail_df = pd.DataFrame(detail_rows)

            # Color rows by Group ID so overlapping trades are visually linked
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
                "showing both reconstructed trades independently, using the selected "
                f"{ratio_option} Mini:Micro ratio."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a fills CSV to begin.")
