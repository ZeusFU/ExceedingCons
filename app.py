import streamlit as st
import pandas as pd

st.title("Overlapping Contracts Checker (by Fills)")

st.write(
    """
    Upload a **fills CSV** (like `fills_export.csv` from Tradovate),
    then enter the **max allowed total simultaneous contracts**.

    The app reconstructs live positions per symbol from individual fills
    and finds time intervals where:
    - You have **positions in at least two different symbols**, and
    - The **sum of live contracts across those symbols** exceeds the max.
    """
)

uploaded_file = st.file_uploader("Upload fills CSV", type=["csv"])

max_contracts = st.number_input(
    "Max allowed simultaneous contracts (across all symbols)",
    min_value=0.0,
    value=5.0,
    step=1.0,
)

def find_overlapping_exposure(df: pd.DataFrame, max_contracts: float):
    """
    From a fills DataFrame with columns:
        - action (Buy/Sell)
        - asset (symbol)
        - quantity
        - timestamp
    return a list of intervals where overlapping exposure across
    different symbols exceeds max_contracts.
    """
    # Make sure the necessary columns are present
    required_cols = {"action", "asset", "quantity", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    # Clean and sort
    df = df.copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["quantity", "timestamp_dt"])
    df = df.sort_values("timestamp_dt").reset_index(drop=True)

    if df.empty:
        return [], df

    # Helper: snapshot current live positions per symbol
    def snapshot_positions(positions):
        # abs() so shorts count as positive exposure
        return {asset: abs(pos) for asset, pos in positions.items() if pos != 0}

    positions = {}  # asset -> net position (signed)
    intervals = []

    # Iterate over fills and examine intervals between them
    for i, row in df.iterrows():
        # For the interval between fill i-1 and fill i, use positions BEFORE applying fill i
        if i > 0:
            start = df.loc[i - 1, "timestamp_dt"]
            end = row["timestamp_dt"]

            snap = snapshot_positions(positions)
            # Active symbols: those with non-zero live contracts
            active_symbols = [s for s, v in snap.items() if v > 0]
            total_open = sum(snap.values())

            # Condition: at least two different symbols AND total > max_contracts
            if len(active_symbols) >= 2 and total_open > max_contracts:
                sym_detail = ", ".join(f"{s}: {snap[s]}" for s in sorted(active_symbols))
                intervals.append(
                    {
                        "Interval Start": start,
                        "Interval End": end,
                        "Total Open Contracts": total_open,
                        "Per-Symbol Positions": sym_detail,
                    }
                )

        # Apply the current fill to update positions
        qty = row["quantity"]
        action = str(row["action"]).strip().lower()
        asset = row["asset"]

        if action == "buy":
            delta = qty
        elif action == "sell":
            delta = -qty
        else:
            # Unknown action type, skip
            continue

        positions[asset] = positions.get(asset, 0) + delta

    # NOTE: After the last fill, positions might still be open.
    # If you want, you could treat the "end" as the last timestamp, but
    # here we only look between known fill times.

    return intervals, df


if uploaded_file is not None:
    try:
        fills_df = pd.read_csv(uploaded_file)

        st.subheader("Raw fills (first 50 rows)")
        st.dataframe(fills_df.head(50))

        intervals, cleaned_df = find_overlapping_exposure(fills_df, max_contracts)

        st.subheader("Cleaned & time-ordered fills used for calculation")
        st.dataframe(
            cleaned_df[["timestamp_dt", "action", "asset", "quantity"]].head(200)
        )

        st.subheader("Overlapping exposure exceeding max contracts")

        if intervals:
            overlaps_df = pd.DataFrame(intervals)
            st.dataframe(overlaps_df)
            st.success(
                f"Found {len(overlaps_df)} interval(s) where overlapping positions "
                f"in different symbols exceeded {max_contracts} contracts."
            )
        else:
            st.info(
                f"No time intervals found where overlapping positions in different "
                f"symbols exceeded {max_contracts} contracts."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a fills CSV to begin.")
