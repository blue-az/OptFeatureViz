# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import gc

# --- Configuration ---
DATA_FILE = "./train.csv" # Default path - can be overridden if needed
# Or use a subdirectory: DATA_FILE = "./data/train.csv"

st.set_page_config(layout="wide") # Use wide layout for more space
st.title("Optiver Trading at the Close - Data Dashboard")

# --- Utility Functions ---
# (Using a simplified reduce_mem_usage for speed in dashboard)
def reduce_mem_usage_dashboard(df, verbose=False):
    """Basic memory reduction for dashboard speed."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            # Downcast numerics
            df[col] = pd.to_numeric(df[col], downcast='integer')
            df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Mem usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    return df

# Cache data loading to speed up app interactions
@st.cache_data
def load_data(filepath):
    """Loads and performs initial processing on the train data."""
    if not os.path.exists(filepath):
        st.error(f"Error: Data file not found at '{filepath}'. Please place train.csv correctly.")
        return None
    try:
        df = pd.read_csv(filepath)
        # Basic cleaning - Competition target is NaN sometimes
        df = df.dropna(subset=['target'])
        df = reduce_mem_usage_dashboard(df)
        # Convert relevant columns if needed (e.g., if date_id isn't int)
        df['date_id'] = df['date_id'].astype(int)
        df['stock_id'] = df['stock_id'].astype(int)
        df['seconds_in_bucket'] = df['seconds_in_bucket'].astype(int)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Load Data ---
df_full = load_data(DATA_FILE)

# Exit if data loading failed
if df_full is None:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Stock Selector
stock_list = sorted(df_full['stock_id'].unique())
selected_stocks = st.sidebar.multiselect(
    "Select Stock(s)",
    options=stock_list,
    default=stock_list[:3] # Default to first 3 stocks
)

# Date Range Selector
min_date, max_date = int(df_full['date_id'].min()), int(df_full['date_id'].max())
selected_date_range = st.sidebar.slider(
    "Select Date ID Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, min_date + 10) # Default to first 10 days
)

# --- Filter Data ---
if not selected_stocks:
    st.warning("Please select at least one stock.")
    df_filtered = pd.DataFrame() # Empty dataframe
else:
    df_filtered = df_full[
        (df_full['stock_id'].isin(selected_stocks)) &
        (df_full['date_id'] >= selected_date_range[0]) &
        (df_full['date_id'] <= selected_date_range[1])
    ].copy() # Use copy to avoid SettingWithCopyWarning later
    gc.collect() # Collect garbage after filtering

st.sidebar.info(f"Displaying data for {len(selected_stocks)} stock(s) between date ID {selected_date_range[0]} and {selected_date_range[1]}.")

# --- Main Dashboard Area ---

if df_filtered.empty and selected_stocks:
     st.warning("No data available for the selected stock(s) and date range.")
elif not df_filtered.empty:
    st.header("Filtered Data Overview")
    st.metric("Number of Rows", f"{len(df_filtered):,}")
    st.dataframe(df_filtered.head())

    # --- Visualizations ---
    st.header("Visualizations")

    # 1. Target Distribution
    with st.expander("Target Distribution (Filtered Data)"):
        fig_target_dist = px.histogram(df_filtered, x='target', nbins=100, title='Distribution of Target Variable')
        fig_target_dist.update_layout(bargap=0.1)
        st.plotly_chart(fig_target_dist, use_container_width=True)
        st.write(df_filtered['target'].describe())

    # 2. Time Series Plot (WAP by default)
    with st.expander("Time Series Plot (Intraday)"):
        ts_cols = ['wap', 'target', 'liquidity_imbalance', 'matched_size', 'volume', 'bid_price', 'ask_price', 'price_spread']
        # Filter only columns that exist in the dataframe
        available_ts_cols = [col for col in ts_cols if col in df_filtered.columns]

        if available_ts_cols:
             selected_ts_col = st.selectbox("Select variable to plot over time", options=available_ts_cols, index=available_ts_cols.index('wap') if 'wap' in available_ts_cols else 0)

             # Performance consideration: Subsample if too many points
             max_points_ts = 50000
             if len(df_filtered) > max_points_ts:
                 st.info(f"Subsampling data to {max_points_ts} points for time series plotting performance.")
                 plot_df = df_filtered.sample(n=max_points_ts).sort_values(by=['date_id', 'seconds_in_bucket'])
             else:
                 plot_df = df_filtered.sort_values(by=['date_id', 'seconds_in_bucket'])

             fig_ts = px.line(plot_df, x='seconds_in_bucket', y=selected_ts_col, color='stock_id',
                              title=f'{selected_ts_col} over Seconds in Bucket (Colored by Stock ID)',
                              hover_data=['date_id'])
             fig_ts.update_layout(xaxis_title='Seconds in Bucket', yaxis_title=selected_ts_col)
             st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning("No suitable columns found for time series plotting.")


    # 3. Scatter Plot (Explore Relationships)
    with st.expander("Relationship Explorer (Scatter Plot)"):
         scatter_cols = ['wap', 'target', 'liquidity_imbalance', 'matched_size', 'volume', 'bid_ask_spread', 'imbalance_size', 'reference_price']
         # Calculate spread if needed
         if 'bid_price' in df_filtered.columns and 'ask_price' in df_filtered.columns:
             df_filtered['bid_ask_spread'] = df_filtered['ask_price'] - df_filtered['bid_price']
             if 'bid_ask_spread' not in scatter_cols: scatter_cols.append('bid_ask_spread')

         available_scatter_cols = [col for col in scatter_cols if col in df_filtered.columns]

         if len(available_scatter_cols) >= 2:
             col1, col2 = st.columns(2)
             with col1:
                 x_axis_val = st.selectbox('Select X-axis variable', options=available_scatter_cols, index=0)
             with col2:
                 y_axis_val = st.selectbox('Select Y-axis variable', options=available_scatter_cols, index=1)

             # Performance consideration: Subsample if too many points
             max_points_scatter = 10000
             if len(df_filtered) > max_points_scatter:
                 st.info(f"Subsampling data to {max_points_scatter} points for scatter plotting performance.")
                 scatter_df = df_filtered.sample(n=max_points_scatter)
             else:
                 scatter_df = df_filtered

             fig_scatter = px.scatter(scatter_df, x=x_axis_val, y=y_axis_val,
                                     color='stock_id', # Optional: color by stock
                                     title=f'Relationship between {x_axis_val} and {y_axis_val}',
                                     hover_data=['date_id', 'seconds_in_bucket'])
             st.plotly_chart(fig_scatter, use_container_width=True)
         else:
             st.warning("Not enough suitable columns found for scatter plotting.")


    # 4. Aggregate View (Box Plot by Stock)
    with st.expander("Aggregate View per Stock (Box Plot)"):
        agg_cols = ['target', 'wap', 'liquidity_imbalance', 'price_spread']
        available_agg_cols = [col for col in agg_cols if col in df_filtered.columns]

        if available_agg_cols:
            selected_agg_col = st.selectbox("Select variable for Box Plot by Stock", options=available_agg_cols, index=0)
            fig_box = px.box(df_filtered, x='stock_id', y=selected_agg_col, color='stock_id',
                             title=f'Distribution of {selected_agg_col} per Stock')
            st.plotly_chart(fig_box, use_container_width=True)
        else:
             st.warning("No suitable columns found for aggregate box plotting.")


    # Add more visualizations as needed...
    # Example: Correlation Heatmap (can be slow)
    # with st.expander("Correlation Heatmap (Sampled Data)"):
    #     corr_cols = ['wap', 'target', 'imbalance_size', 'matched_size', 'liquidity_imbalance', 'volume']
    #     available_corr_cols = [c for c in corr_cols if c in df_filtered.columns]
    #     if len(available_corr_cols) > 1:
    #         max_points_corr = 5000
    #         if len(df_filtered) > max_points_corr:
    #             corr_df = df_filtered.sample(n=max_points_corr)[available_corr_cols]
    #         else:
    #             corr_df = df_filtered[available_corr_cols]
    #         corr = corr_df.corr()
    #         fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap (Sampled Data)")
    #         st.plotly_chart(fig_corr, use_container_width=True)


else:
    st.info("Select stocks and date range from the sidebar to begin analysis.")

st.markdown("---")
st.markdown("Dashboard by Enhanced Average Solution Analysis")
