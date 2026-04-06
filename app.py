import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Historical GPU Statistical Analysis", layout="wide")

st.title("🖥️ Historical GPU Statistical Analysis")
st.markdown("This application explores the relationships between GPU specifications and their pricing history.")

# --- DATA LOADING & CONTEXTUALIZATION ---
@st.cache_data
def load_data():
    """
    Loads and preprocesses the GPU datasets.
    It expects two files: 'gpu_metadata.csv' and 'gpu_price_history.csv' 
    in the same directory, or you can update the paths below.
    """
    try:
        # PLACEHOLDER: Update these filenames if your actual files are named differently
        meta_filepath = 'gpu_metadata.csv'
        price_filepath = 'gpu_price_history.csv'
        
        df_meta = pd.read_csv(meta_filepath)
        df_price = pd.read_csv(price_filepath)

        # Preprocessing metadata
        # Extract Brand based on naming convention
        df_meta['Brand'] = df_meta['Name'].apply(lambda x: 'NVIDIA' if 'GeForce' in str(x) else ('AMD' if 'Radeon' in str(x) else 'Other'))
        
        # Clean 'Wattage' (e.g. "75W" -> 75) and 'VRAM' (e.g. "2GB" -> 2)
        df_meta['Wattage'] = df_meta['Wattage'].astype(str).str.replace('W', '', regex=False).astype(float)
        df_meta['VRAM'] = df_meta['VRAM'].astype(str).str.replace('GB', '', regex=False).astype(float)
        df_meta.rename(columns={'Wattage': 'Watt Rating', '3DMARK': '3DMark Benchmark Value'}, inplace=True)

        # Preprocessing pricing data
        # To get a single value per GPU, we'll take the mean of historical prices
        # You could modify this to take the latest price, max price, etc.
        # Replacing 0 with NaN so it doesn't skew the mean artificially 
        df_price['Retail Price'] = df_price['Retail Price'].replace(0, np.nan)
        df_price['Used Price'] = df_price['Used Price'].replace(0, np.nan)
        # Parse Dates for Time Series
        df_price['Date'] = pd.to_datetime(df_price['Date'], format='%d-%m-%y', errors='coerce')
        
        df_price_avg = df_price.groupby('Name')[['Retail Price', 'Used Price']].mean().reset_index()
        df_price_avg.rename(columns={'Retail Price': 'Historical Retail Price', 'Used Price': 'Historical Used Price'}, inplace=True)

        # Merge metadata and average pricing
        df = pd.merge(df_meta, df_price_avg, on='Name', how='inner')
        
        return df, df_price

    except FileNotFoundError:
        st.error(f"Error: Could not find the dataset files. Please ensure '{meta_filepath}' and '{price_filepath}' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None, None

df, df_price = load_data()

# Load contextualization text
context_file = 'Statistical Contextualization of GPUs.txt'
if os.path.exists(context_file):
    with open(context_file, 'r', encoding='utf-8') as f:
        context_text = f.read()
else:
    context_text = "> *Context file not found. Place 'Statistical Contextualization of GPUs.txt' in the directory.*"

if df is not None:
    # --- NAVIGATION TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Context & Data Overview", 
        "Descriptive Statistics", 
        "Regression Analysis", 
        "Chi-Square Testing", 
        "ANOVA Testing",
        "Time Series Analysis"
    ])

    # --- 1. CONTEXT & DATA OVERVIEW ---
    with tab1:
        st.header("1. Data Loading & Contextualization")
        with st.expander("Read Contextual Background", expanded=True):
            st.markdown(context_text)
        
        st.subheader("Raw Data Preview")
        st.dataframe(df)

    # --- 2. DESCRIPTIVE STATISTICS ---
    with tab2:
        st.header("2. Descriptive Statistics")
        st.markdown("Summary of continuous variables in the dataset.")
        
        # UI for Filtering
        gpu_list = sorted(df['Name'].unique())
        selected_gpus = st.multiselect("Select Models to Include:", gpu_list, default=gpu_list)
        
        min_date = df_price['Date'].min().date()
        max_date = df_price['Date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date (Price Window)", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date (Price Window)", value=max_date, min_value=min_date, max_value=max_date)
            
        if start_date > end_date:
            st.error("Error: Start date must fall before end date.")
        elif not selected_gpus:
            st.warning("Please select at least one GPU model.")
        else:
            # Step 1: Filter raw prices by date
            mask = (df_price['Date'].dt.date >= start_date) & (df_price['Date'].dt.date <= end_date)
            filtered_price = df_price[mask]
            
            # Step 2: Extract real price points bounded by selection and merge
            if not filtered_price.empty:
                # Fetch base metadata for selected GPUs (stripping out the global averages)
                meta_subset = df[df['Name'].isin(selected_gpus)].drop(columns=['Historical Retail Price', 'Historical Used Price'], errors='ignore')
                
                # Fetch raw chronological prices for exactly the chosen models
                price_subset = filtered_price[filtered_price['Name'].isin(selected_gpus)]
                
                # Re-merge the raw time-series prices so standard deviation and min/max reflect price shifts
                custom_df = pd.merge(meta_subset, price_subset, on='Name', how='inner')
                custom_df.rename(columns={'Retail Price': 'Historical Retail Price', 'Used Price': 'Historical Used Price'}, inplace=True)
            else:
                custom_df = df[df['Name'].isin(selected_gpus)].copy()
                custom_df['Historical Retail Price'] = np.nan
                custom_df['Historical Used Price'] = np.nan

            continuous_vars = ['VRAM', 'Watt Rating', '3DMark Benchmark Value', 'Historical Retail Price', 'Historical Used Price']
            
            # Calculate descriptive stats
            desc_stats = custom_df[continuous_vars].describe().T
            desc_stats['median'] = custom_df[continuous_vars].median()
            # Reorder columns for clean presentation
            desc_stats = desc_stats[['mean', 'median', 'std', 'min', 'max']]
            
            st.table(desc_stats)

    # --- 3. REGRESSION ANALYSIS ---
    with tab3:
        st.header("3. Interactive Regression Analysis")
        
        continuous_vars = ['VRAM', 'Watt Rating', '3DMark Benchmark Value', 'Historical Retail Price', 'Historical Used Price']
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Select Independent Variable (X)", continuous_vars, index=2) # Default to 3DMark
        with col2:
            y_var = st.selectbox("Select Dependent Variable (Y)", continuous_vars, index=3) # Default to Retail Price

        if x_var and y_var:
            # Drop NaNs for regression 
            reg_df = df.dropna(subset=[x_var, y_var])
            
            if len(reg_df) > 1:
                # Scatter Plot
                fig = px.scatter(reg_df, x=x_var, y=y_var, color='Brand', hover_data=['Name'], 
                                 trendline="ols", trendline_color_override="red",
                                 title=f"Regression: {x_var} vs {y_var}")
                st.plotly_chart(fig, use_container_width=True)

                # Regression Statistics via statsmodels
                X = sm.add_constant(reg_df[x_var])
                Y = reg_df[y_var]
                model = sm.OLS(Y, X).fit()
                
                intercept = model.params['const']
                slope = model.params[x_var]
                r_squared = model.rsquared
                p_value = model.pvalues[x_var]

                st.subheader("Regression Statistics")
                st.write(f"**Equation:** `Y = {slope:.4f} * X + {intercept:.4f}`")
                st.write(f"**R-squared:** `{r_squared:.4f}`")
                st.write(f"**P-value:** `{p_value:.4e}`")
                
                if p_value < 0.05:
                    st.success("The predictor variable is statistically significant (p < 0.05).")
                else:
                    st.warning("The predictor variable is NOT statistically significant (p >= 0.05).")
            else:
                st.warning("Not enough continuous valid data points to perform regression.")

    # --- 4. CHI-SQUARE TESTING ---
    with tab4:
        st.header("4. Chi-Square Testing of Independence")
        st.markdown("Tests if there is a significant association between GPU Brand and a binned continuous variable.")

        target_var = st.selectbox("Select Continuous Variable to Bin:", 
                                  ['Historical Retail Price', '3DMark Benchmark Value', 'Watt Rating'], 
                                  index=0)
        
        num_bins = st.slider("Number of Categories (Bins)", min_value=2, max_value=5, value=3)

        # Drop NaNs first
        chi_df = df.dropna(subset=['Brand', target_var]).copy()

        if len(chi_df) > 0:
            # Binning logic (using qcut for quantiles, fallback to cut if edges are non-unique)
            labels = ["Low", "Medium", "High", "Very High", "Extreme"][:num_bins]
            try:
                chi_df[f'{target_var}_Binned'] = pd.qcut(chi_df[target_var], q=num_bins, labels=labels, duplicates='drop')
            except Exception:
                chi_df[f'{target_var}_Binned'] = pd.cut(chi_df[target_var], bins=num_bins, labels=labels)
            
            # Contingency table
            contingency_table = pd.crosstab(chi_df['Brand'], chi_df[f'{target_var}_Binned'])
            
            st.subheader("Contingency Table")
            st.dataframe(contingency_table)

            # Perform test
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            
            st.subheader("Chi-Square Results")
            st.write(f"**Chi-Square Statistic:** `{chi2:.4f}`")
            st.write(f"**P-value:** `{p_val:.4e}`")
            st.write(f"**Degrees of Freedom:** `{dof}`")

            if p_val < 0.05:
                st.success(f"**Interpretation:** There is a statistically significant association between GPU Brand and {target_var}.")
            else:
                st.info(f"**Interpretation:** We fail to reject the null hypothesis. There is no significant association between GPU Brand and {target_var}.")
        else:
            st.warning("Not enough data to perform Chi-Square test.")

    # --- 5. ANOVA TESTING ---
    with tab5:
        st.header("5. One-Way ANOVA Testing")
        st.markdown("Tests if there is a statistically significant difference in variance between GPU Brands for a given metric.")

        anova_var = st.selectbox("Select metric to test between NVIDIA and AMD:", 
                                  ['Historical Retail Price', '3DMark Benchmark Value', 'Historical Used Price'], 
                                  index=0)

        # Prepare groups
        grp_nvidia = df[df['Brand'] == 'NVIDIA'][anova_var].dropna()
        grp_amd = df[df['Brand'] == 'AMD'][anova_var].dropna()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("NVIDIA GPUs Count", len(grp_nvidia))
            st.metric(f"NVIDIA Mean {anova_var}", f"{grp_nvidia.mean():.2f}")
        with col2:
            st.metric("AMD GPUs Count", len(grp_amd))
            st.metric(f"AMD Mean {anova_var}", f"{grp_amd.mean():.2f}")

        if len(grp_nvidia) > 0 and len(grp_amd) > 0:
            # Perform ANOVA
            f_stat, p_val = stats.f_oneway(grp_nvidia, grp_amd)

            st.subheader("ANOVA Results")
            st.write(f"**F-statistic:** `{f_stat:.4f}`")
            st.write(f"**P-value:** `{p_val:.4e}`")

            if p_val < 0.05:
                st.success(f"**Interpretation:** There is a statistically significant difference in {anova_var} between NVIDIA and AMD GPUs.")
            else:
                st.info(f"**Interpretation:** There is NO statistically significant difference in {anova_var} between NVIDIA and AMD GPUs.")
            
            # Optional Boxplot for visual confirmation
            st.subheader("Distribution Visualization")
            fig = px.box(df.dropna(subset=[anova_var]), x="Brand", y=anova_var, color="Brand", points="all")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to run the ANOVA test. Ensure both AMD and NVIDIA have valid records.")

    # --- 6. TIME SERIES ANALYSIS ---
    with tab6:
        st.header("6. Price Time Series Analysis")
        st.markdown("Visualize how the Retail and Used prices of GPUs have changed over time.")

        # Let the user select multiple GPUs to compare (sorting names for clean UI)
        gpu_list = sorted(df_price['Name'].dropna().unique())
        if len(gpu_list) > 0:
            selected_gpus = st.multiselect("Select GPUs to plot:", gpu_list, default=[gpu_list[0]])
            
            price_type = st.radio("Select Price Type:", ["Retail Price", "Used Price"], horizontal=True)

            if selected_gpus:
                ts_df = df_price[df_price['Name'].isin(selected_gpus)].copy()
                # Sort by date
                ts_df = ts_df.sort_values(by='Date')

                fig = px.line(ts_df, x='Date', y=price_type, color='Name', markers=True,
                              title=f"Historical {price_type} Over Time",
                              labels={price_type: f"Price (USD)", "Date": "Date"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one GPU to visualize.")
        else:
            st.warning("No GPU names found in the datasets to track.")

else:
    st.info("Awaiting data load to initialize analysis sections.")
