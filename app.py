import streamlit as st
import pandas as pd

st.title("Overlapping Contracts Checker (by Fills, Color-Coded)")

st.write(
    """
    Upload a **fills CSV** (like `fills_export.csv`), then enter the 
    **max allowed total simultaneous contracts across all symbols**.

    The app:
    1. Reconstructs live positions per symbol from the fills.
    2. Finds time intervals where:
       - At least **two different symbols** are live, and
       - The **sum of live contracts** across them exceeds the max.
    3. Shows:
       - A summary of these intervals.
       - A **color-coded detailed table** where each symbol's exposure
         in each interval is shown independently, with rows belonging to
         the same interval shaded the same color.
    """
)

uploaded_file = st.file_uploader("Upload fills CSV", type=["csv"])

max_contracts = st.number_input(
    "Max allowed simultaneous contracts (across all symbols)",
    min_value=0.0,
    value=5.0,
    step=1.0,
)


def find_overlapping_exposure_with_details(df: pd.DataFrame, max_contracts: float):
    """
    From a fills DataFrame with columns:
        - action (Buy/Sell)
        - asset (symbol)
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

    def snapshot_positions(positions):
        # abs() so shorts count as positive exposure
        return {asset: abs(pos) for asset, pos in positions.items() if pos != 0}

    positions = {}  # asset -> net position (signed)
    intervals_summary = []
    detail_rows = []
    group_id = 0

    for i, row in df.iterrows():
        # Interval between fill i-1 and fill i uses positions BEFORE applying fill i
        if i > 0:
            start = df.loc[i - 1, "timestamp_dt"]
            end = row["timestamp_dt"]

            snap = snapshot_positions(positions)
            active_symbols = [s for s, v in snap.items() if v > 0]
            total_open = sum(snap.values())

            if len(active_symbols) >= 2 and total_open > max_contracts:
                group_id += 1

                # Summary row (one per interval)
                intervals_summary.append(
                    {
                        "Group ID": group_id,
                        "Interval Start": start,
                        "Interval End": end,
                        "Total Open Contracts": total_open,
                        "Active Symbols": ", ".join(
                            f"{s}:{snap[s]}" for s in sorted(active_symbols)
                        ),
                    }
                )

                # Detailed rows (one per symbol per interval)
                for sym in sorted(active_symbols):
                    detail_rows.append(
                        {
                            "Group ID": group_id,
                            "Interval Start": start,
                            "Interval End": end,
                            "Symbol": sym,
                            "Open Contracts": snap[sym],
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
            # If we have unknown actions, skip them
            continue

        positions[asset] = positions.get(asset, 0) + delta

    return intervals_summary, detail_rows, df


if uploaded_file is not None:
    try:
        fills_df = pd.read_csv(uploaded_file)

        st.subheader("Raw fills (first 50 rows)")
        st.dataframe(fills_df.head(50))

        intervals_summary, detail_rows, cleaned_df = find_overlapping_exposure_with_details(
            fills_df, max_contracts
        )

        st.subheader("Cleaned & time-ordered fills used for calculation")
        st.dataframe(
            cleaned_df[["timestamp_dt", "action", "asset", "quantity"]].head(200),
            use_container_width=True,
        )

        st.subheader("Overlapping intervals summary")

        if not intervals_summary:
            st.info(
                f"No time intervals found where overlapping positions in different "
                f"symbols exceeded {max_contracts} contracts."
            )
        else:
            summary_df = pd.DataFrame(intervals_summary)
            st.dataframe(summary_df, use_container_width=True)

            st.success(
                f"Found {len(summary_df)} interval(s) where overlapping positions "
                f"in different symbols exceeded {max_contracts} contracts."
            )

            st.subheader("Detailed per-symbol overlapping exposure (color-coded)")

            detail_df = pd.DataFrame(detail_rows)

            # Assign a color to each Group ID so that rows from the same
            # overlapping interval share the same background.
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
                gid: color_cycle[(idx % len(color_cycle))]
                for idx, gid in enumerate(unique_groups)
            }
            detail_df["Color"] = detail_df["Group ID"].map(group_to_color)

            # Use a Styler to color-code the rows
            def color_rows(row):
                color = row["Color"]
                return [
                    f"background-color: {color}" if col != "Color" else ""
                    for col in row.index
                ]

            styled_detail = detail_df.style.apply(color_rows, axis=1)

            st.dataframe(styled_detail, use_container_width=True)

            st.caption(
                "Each color represents one interval where total open contracts across "
                "multiple symbols exceeded the max. Rows with the same color belong "
                "to the same overlapping interval, showing each symbol's reconstructed "
                "position independently."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a fills CSV to begin.")
