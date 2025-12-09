import streamlit as st
import pandas as pd

st.title("Overlapping Trades Volume Checker")

st.write(
    """
    Upload a trades CSV (like your `trades_export (22).csv`), 
    then enter the **max total overlapping volume**.
    
    The app will find **pairs of trades** with **different Symbols** that overlap in time
    and whose combined Volume exceeds that max.
    """
)

uploaded_file = st.file_uploader("Upload trades CSV", type=["csv"])

max_volume = st.number_input(
    "Max total overlapping volume",
    min_value=0.0,
    value=5.0,
    step=1.0,
)

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Basic sanity check
    required_cols = {"Symbol", "Volume", "Open Time", "Close Time"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(missing)}")
    else:
        # Ensure Volume is numeric
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

        # Parse times
        df["Open_dt"] = pd.to_datetime(df["Open Time"], errors="coerce")
        df["Close_dt"] = pd.to_datetime(df["Close Time"], errors="coerce")

        # Drop rows with invalid times or volume
        df_clean = df.dropna(subset=["Open_dt", "Close_dt", "Volume"]).copy()

        if df_clean.empty:
            st.warning("No valid rows after cleaning (check date/time and volume fields).")
        else:
            st.subheader("Input Trades (cleaned)")
            st.dataframe(df_clean[["Symbol", "Volume", "Open Time", "Close Time"]])

            # Find overlapping trades with different symbols whose total volume > max_volume
            overlaps = []

            n = len(df_clean)
            for i in range(n):
                for j in range(i + 1, n):
                    row_i = df_clean.iloc[i]
                    row_j = df_clean.iloc[j]

                    # Only consider different symbols
                    if row_i["Symbol"] == row_j["Symbol"]:
                        continue

                    # Compute overlap
                    latest_start = max(row_i["Open_dt"], row_j["Open_dt"])
                    earliest_end = min(row_i["Close_dt"], row_j["Close_dt"])

                    # Strict overlap: start < end
                    if latest_start < earliest_end:
                        total_vol = row_i["Volume"] + row_j["Volume"]

                        if total_vol > max_volume:
                            overlaps.append(
                                {
                                    "Symbol 1": row_i["Symbol"],
                                    "Volume 1": row_i["Volume"],
                                    "Open Time 1": row_i["Open Time"],
                                    "Close Time 1": row_i["Close Time"],
                                    "Symbol 2": row_j["Symbol"],
                                    "Volume 2": row_j["Volume"],
                                    "Open Time 2": row_j["Open Time"],
                                    "Close Time 2": row_j["Close Time"],
                                    "Overlap Start": latest_start,
                                    "Overlap End": earliest_end,
                                    "Total Volume": total_vol,
                                }
                            )

            st.subheader("Overlapping Trades Exceeding Max Volume")

            if overlaps:
                overlaps_df = pd.DataFrame(overlaps)
                st.dataframe(overlaps_df)
                st.success(
                    f"Found {len(overlaps)} pair(s) of overlapping trades with different Symbols "
                    f"whose total Volume exceeds {max_volume}."
                )
            else:
                st.info(
                    f"No overlapping trades with different Symbols exceed a total Volume of {max_volume}."
                )
else:
    st.info("Upload a CSV file to begin.")
