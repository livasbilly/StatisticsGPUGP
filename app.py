import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import os
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

LABEL_MAP = {
    'Historical Retail Price': 'Historical Retail Price (USD)',
    'Historical Used Price': 'Historical Used Price (USD)',
    'Retail Price': 'Retail Price (USD)',
    'Used Price': 'Used Price (USD)',
    'Watt Rating': 'Watt Rating (W)',
    'VRAM': 'VRAM (GB)'
}

if df is not None:
    if 'VRAM' in df.columns:
        df['VRAM Tier'] = df['VRAM'].apply(lambda v: f"{int(v)}GB" if pd.notna(v) and float(v).is_integer() else f"{v}GB" if pd.notna(v) else np.nan)

    # --- NAVIGATION TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Context & Data Overview", 
        "Descriptive Statistics", 
        "Regression Analysis", 
        "Chi-Square Testing", 
        "ANOVA Testing",
        "Time Series Analysis",
        "Q&A"
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
                # Scatter Plot (No default trendline)
                fig = px.scatter(reg_df, x=x_var, y=y_var, color='Brand', hover_data=['Name'], 
                                 title=f"Regression: {y_var} vs {x_var}", labels=LABEL_MAP)

                # Regression Statistics via statsmodels
                X_vals = reg_df[x_var]
                Y_vals = reg_df[y_var]

                # Regression 1: Y on X (minimizes vertical errors)
                X1 = sm.add_constant(X_vals)
                model1 = sm.OLS(Y_vals, X1).fit()
                c1 = model1.params['const']
                m1 = model1.params[x_var]

                # Regression 2: X on Y (minimizes horizontal errors)
                Y2 = sm.add_constant(Y_vals)
                model2 = sm.OLS(X_vals, Y2).fit()
                c2 = model2.params['const']
                m2 = model2.params[y_var]

                # Convert Eq 2 to Y = mX + c format for plotting on the same Y vs X axes
                # X = m2 * Y + c2  =>  m2 * Y = X - c2  =>  Y = (1/m2) * X - (c2/m2)
                m2_plot = 1 / m2
                c2_plot = -c2 / m2

                # Add regression lines to the figure
                x_range = np.array([X_vals.min(), X_vals.max()])
                
                y_range_1 = m1 * x_range + c1
                fig.add_trace(go.Scatter(x=x_range, y=y_range_1, mode='lines', 
                                         name='Y on X (Vertical Errors)', 
                                         line=dict(color='red', width=2)))
                
                y_range_2 = m2_plot * x_range + c2_plot
                fig.add_trace(go.Scatter(x=x_range, y=y_range_2, mode='lines', 
                                         name='X on Y (Horizontal Errors)', 
                                         line=dict(color='blue', width=2, dash='dash')))

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Regression Statistics")
                
                st.markdown(f"**Equation 1 (Y on X):** `Y = {m1:.4f} * X {c1:+.4f}` *(minimizes vertical errors)*")
                st.markdown(f"**Equation 2 (X on Y):** `X = {m2:.4f} * Y {c2:+.4f}` &nbsp; $\\Rightarrow$ &nbsp; plotted as `Y = {m2_plot:.4f} * X {c2_plot:+.4f}` *(minimizes horizontal errors)*")
                
                r_squared_calc = m1 * m2
                p_value = model1.pvalues[x_var]

                st.write(f"**R-squared ($R^2$):** `{r_squared_calc:.4f}` *(Calculated as the product of the two slopes: {m1:.4f} * {m2:.4f})*")
                st.write(f"**P-value:** `{p_value:.4e}`")
                
                if p_value < 0.05:
                    st.success("The predictor variable is statistically significant (p < 0.05).")
                else:
                    st.warning("The predictor variable is NOT statistically significant (p >= 0.05).")
            else:
                st.warning("Not enough continuous valid data points to perform regression.")

    # --- 4. CHI-SQUARE TESTING ---
    with tab4:
        st.header("4. Chi-Square Testing of Independence", help="**The 'Expectation vs. Reality' Test (for categories)**\n\n* **What it does:** It checks for relationships between *categorical* variables (things you count in buckets, like \"Brand: AMD vs NVIDIA\" or \"Color: Red vs Blue\").\n* **How it works:** It compares the actual number of items you counted in each category against the number you *expected* to see if everything was completely random. If the actual counts wildly differ from the expected counts, there is a statistical relationship.")
        st.markdown("Tests if there is a significant association between two categorical variables.")

        cats = ['Brand', 'VRAM Tier']
        conts = ['Historical Retail Price', 'Historical Used Price', 'Watt Rating', '3DMark Benchmark Value']
        all_vars = cats + conts

        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Select Variable 1 (Rows):", all_vars, index=0)
        with col2:
            var2 = st.selectbox("Select Variable 2 (Columns):", all_vars, index=1)

        if var1 == var2:
            st.warning("Please select two distinct variables to perform the Chi-Square test.")
        else:
            col_b1, col_b2 = st.columns(2)
            bins_var1 = None
            bins_var2 = None

            if var1 in conts:
                with col_b1:
                    bins_var1 = st.slider(f"Number of Categories for {var1}", min_value=2, max_value=5, value=3, key='bins1')
            if var2 in conts:
                with col_b2:
                    bins_var2 = st.slider(f"Number of Categories for {var2}", min_value=2, max_value=5, value=3, key='bins2')

            chi_df = df.dropna(subset=[var1, var2]).copy()

            import re
            def alphanumeric_key(val):
                s = str(val)
                match = re.match(r'^(\d+)', s)
                if match:
                    return (0, int(match.group(1)), s)
                return (1, 0, s)

            def render_grouping_tool(var_name):
                st.subheader(f"Category Grouping Tool: {var_name}")
                st.markdown(f"Assign raw '{var_name}' categories into predefined parent buckets.")
                
                raw_options = sorted(chi_df[var_name].unique().tolist(), key=alphanumeric_key)
                assigned_options = set()
                buckets = {}
                
                predefined_groups = []
                if var_name == 'Brand':
                    predefined_groups = ['AMD', 'NVIDIA']
                elif var_name == 'VRAM Tier':
                    predefined_groups = ['Budget Floor', 'Enthusiast Core', 'Halo']
                else:
                    return var_name

                for i, bucket_name in enumerate(predefined_groups):
                    col_name, col_sel = st.columns([1, 2])
                    col_name.markdown(f"<p style='margin-top: 10px'><b>{bucket_name}</b></p>", unsafe_allow_html=True)
                    # Label is required but visually collapsed to align properly
                    selected = col_sel.multiselect(f"Assign items to {bucket_name}", raw_options, key=f"sel_{var_name}_{i}", label_visibility="collapsed")
                    
                    if selected:
                        buckets[bucket_name] = selected
                        assigned_options.update(selected)
                
                unassigned = [opt for opt in raw_options if opt not in assigned_options]
                if unassigned:
                    st.markdown(f"**Unassigned Pool:** `{', '.join([str(x) for x in unassigned])}`")
                else:
                    st.markdown("**Unassigned Pool:** *All items assigned!*")
                
                if buckets:
                    bucket_map = {item: k for k, v in buckets.items() for item in v}
                    col_id = f'{var_name} (Grouped)'
                    chi_df[col_id] = chi_df[var_name].apply(lambda x: bucket_map.get(x, x))
                    return col_id
                return var_name

            var1_col = var1
            var2_col = var2

            if var1 in cats:
                var1_col = render_grouping_tool(var1)
                st.divider()
            if var2 in cats:
                var2_col = render_grouping_tool(var2)
                st.divider()

            if len(chi_df) > 0:
                # Binning logic
                def bin_variable(dataframe, var_name, num_bins):
                    labels = ["Low", "Medium", "High", "Very High", "Extreme"][:num_bins]
                    try:
                        return pd.qcut(dataframe[var_name], q=num_bins, labels=labels, duplicates='drop')
                    except Exception:
                        return pd.cut(dataframe[var_name], bins=num_bins, labels=labels)

                var1_final = f'{var1}_Binned' if var1 in conts else var1_col
                var2_final = f'{var2}_Binned' if var2 in conts else var2_col

                if var1 in conts:
                    chi_df[var1_final] = bin_variable(chi_df, var1, bins_var1)
                if var2 in conts:
                    chi_df[var2_final] = bin_variable(chi_df, var2, bins_var2)

                # Remove any nan bins
                chi_df = chi_df.dropna(subset=[var1_final, var2_final])

                if len(chi_df) > 0:
                    contingency_table = pd.crosstab(chi_df[var1_final], chi_df[var2_final])
                    
                    # Apply Alphanumeric Sort for categorical axes manually
                    if var1 in cats:
                        sorted_idx = sorted(contingency_table.index, key=alphanumeric_key)
                        contingency_table = contingency_table.reindex(index=sorted_idx)
                    if var2 in cats:
                        sorted_cols = sorted(contingency_table.columns, key=alphanumeric_key)
                        contingency_table = contingency_table.reindex(columns=sorted_cols)

                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        st.warning("Not enough distinct categories for testing. Try adjusting the bin counts/groups.")
                    else:
                        # Perform test
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                        expected_table = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
                        
                        st.subheader("Chi-Square Results")
                        st.write(f"**Chi-Square Statistic:** `{chi2:.4f}`")
                        st.write(f"**Degrees of Freedom:** `{dof}`")
                        st.write(f"**P-value:** `{p_val:.4e}`")

                        if p_val < 0.05:
                            st.success(f"**Interpretation:** There is a statistically significant association between {var1_col} and {var2_col}.")
                        else:
                            st.info(f"**Interpretation:** We fail to reject the null hypothesis. There is no significant association between {var1_col} and {var2_col}.")

                        st.subheader("Contingency Table")
                        table_view = st.radio("Select View:", ["Observed Counts", "Expected Counts"], horizontal=True)
                        if table_view == "Observed Counts":
                            st.dataframe(contingency_table, use_container_width=True)
                        else:
                            st.dataframe(expected_table.round(2), use_container_width=True)
                                
                        # Standardized Residuals Heatmap
                        st.subheader("Heatmap of Standardized Residuals")
                        st.markdown("Colors represent the deviation of observed counts from expected counts. Large positive values (blue) mean more observations than expected; large negative values (red) mean fewer observations than expected.")
                        residuals = (contingency_table - expected_table) / np.sqrt(expected_table)
                        
                        fig = px.imshow(residuals, 
                                        text_auto=".2f", 
                                        aspect="auto",
                                        color_continuous_scale="RdBu",
                                        color_continuous_midpoint=0,
                                        labels=dict(x=var2_col, y=var1_col, color="Residual"))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough valid binned data to perform Chi-Square test.")
            else:
                st.warning("Not enough valid data points for selected variables.")

    # --- 5. STATISTICAL DIFFERENCE TESTING (MULTI-GROUP & POST-HOC) ---
    with tab5:
        st.header("5. Statistical Difference Testing")
        st.markdown("Tests if there is a statistically significant difference between groupings for a given metric. Automatically routes to the appropriate test (ANOVA, Kruskal-Wallis, Mann-Whitney U) based on data normality and group counts, and performs Post-Hoc analysis if required.")

        # 1. Independent Variable Selection
        group_var = st.selectbox("Select Grouping Variable (Independent):", ['Brand', 'VRAM Tier'], index=0)
        
        # Determine valid dependent metrics
        all_metrics = ['Historical Retail Price', 'Historical Used Price', '3DMark Benchmark Value', 'Watt Rating', 'VRAM']
        if group_var == 'VRAM Tier':
            valid_metrics = [m for m in all_metrics if m != 'VRAM']
        else:
            valid_metrics = all_metrics
            
        anova_var = st.selectbox("Select Metric (Dependent):", valid_metrics, index=0)

        # 2. Data Preparation based on independent variable
        if group_var == 'Brand':
            plot_x = 'Brand'
            df_test = df.dropna(subset=[plot_x, anova_var]).copy()
            unique_groups = sorted(df_test[plot_x].unique())
            selected_groups = st.multiselect("Select Brands to Include:", unique_groups, default=unique_groups)
        else:
            plot_x = 'VRAM'
            df_test = df.dropna(subset=[plot_x, anova_var]).copy()
            
            unique_vram_raw = sorted(df_test['VRAM'].unique())
            vram_label_map = {v: f"{int(v)}GB" if v.is_integer() else f"{v}GB" for v in unique_vram_raw}
            
            selected_vram_labels = st.multiselect("Select VRAM Tiers to Include:", list(vram_label_map.values()), default=list(vram_label_map.values()))
            
            selected_groups_raw = [k for k, v in vram_label_map.items() if v in selected_vram_labels]
            df_test = df_test[df_test['VRAM'].isin(selected_groups_raw)].copy()
            df_test['VRAM Tier'] = df_test['VRAM'].map(vram_label_map)
            plot_x = 'VRAM Tier'
            selected_groups = selected_vram_labels

        if not selected_groups or len(selected_groups) < 2:
            st.warning("Please select at least two distinct groups to compare.")
        else:
            # Enforce strict sensible ordering (alphabetical for Brand, numeric for VRAM)
            if group_var == 'Brand':
                selected_groups = sorted(selected_groups)
                df_test = df_test[df_test[plot_x].isin(selected_groups)]
            else:
                base_order = list(vram_label_map.values())
                selected_groups = sorted(selected_groups, key=lambda x: base_order.index(x) if x in base_order else 0)
                
            # Show all metrics regardless of count so missing categories are clearly visible
            # We break this into a grid (max 4 columns per row) to prevent text truncation (...)
            max_cols_per_row = 4
            for i in range(0, len(selected_groups), max_cols_per_row):
                chunk = selected_groups[i:i + max_cols_per_row]
                cols = st.columns(max_cols_per_row)
                for j, g in enumerate(chunk):
                    g_data = df_test[df_test[plot_x] == g][anova_var]
                    with cols[j]:
                        with st.container(border=True):
                            st.metric(f"{g} Count", len(g_data))
                            if len(g_data) > 0:
                                st.metric(f"{g} Mean", f"{g_data.mean():.2f}")
                            else:
                                st.metric(f"{g} Mean", "N/A")

            # Filter for rigorous statistical testing
            valid_groups = [g for g in selected_groups if len(df_test[df_test[plot_x] == g][anova_var]) >= 3]
            excluded_groups = [g for g in selected_groups if g not in valid_groups]
            
            if excluded_groups:
                st.warning(f"Note: Groups with insufficient data (n < 3) were excluded from statistical testing and plots: {', '.join(excluded_groups)}")

            groups_data = {g: df_test[df_test[plot_x] == g][anova_var] for g in valid_groups}
            valid_group_count = len(valid_groups)
            
            # Filter df_test so plots match the tests exactly (excluded items are completely dropped)
            df_test = df_test[df_test[plot_x].isin(valid_groups)]

            if valid_group_count < 2:
                st.warning("Not enough data. Ensure at least two selected groups have $\ge$ 3 valid records.")
            else:
                # Step A: Assumption Check (Normality)
                st.markdown("##### 🔬 Normality Check: Shapiro-Wilk Test", help="**The 'Bell Curve' Test**\n\n* **What it does:** It checks if your data follows a normal distribution (a symmetrical, bell-shaped curve).\n* **When to use it:** You run this *before* other tests. Many tests (like ANOVA) require your data to be normal. If Shapiro-Wilk says your data is *not* normal (skewed), you have to use alternative, \"non-parametric\" tests (like Mann-Whitney).")
                normality_p_values = []
                for g_data in groups_data.values():
                    _, p_val = stats.shapiro(g_data)
                    normality_p_values.append(p_val)
                    
                shapiro_p = min(normality_p_values)
                is_normal = shapiro_p >= 0.05
                
                # Step B: Test Selection
                data_series = list(groups_data.values())
                perform_posthoc = False
                posthoc_type = None
                
                if is_normal and valid_group_count == 2:
                    st.markdown("🟢 **Test Used: One-Way ANOVA (2 Groups)**", help="**The 'Big Picture' Test (for 3+ groups)**\n\n* **What it does:** It compares the averages (means) of three or more groups to see if at least one group is statistically different from the others.\n* **The Catch:** It tells you *that* a difference exists somewhere, but it does not tell you *which* specific groups are different.")
                    stat, p_val = stats.f_oneway(*data_series)
                    st.write(f"**F-statistic:** `{stat:.4f}` | **P-value:** `{p_val:.4e}`")
                    
                elif not is_normal and valid_group_count == 2:
                    st.markdown("🟡 **Test Used: Mann-Whitney U**", help="**The 'Ranked Match' Test (for 2 groups)**\n\n* **What it does:** It compares two groups when your data is messy, has extreme outliers, or failed the Shapiro-Wilk test (meaning it isn't a bell curve).\n* **How it works:** Instead of comparing averages, it lumps all the data from both groups together, ranks them from lowest to highest, and checks if one group's values consistently rank higher than the other.")
                    st.warning("Data Distribution Alert: Your data failed normality. Automatically switched to Mann-Whitney U.")
                    stat, p_val = stats.mannwhitneyu(data_series[0], data_series[1], alternative='two-sided')
                    st.write(f"**U-statistic:** `{stat:.4f}` | **P-value:** `{p_val:.4e}`")
                    
                elif is_normal and valid_group_count >= 3:
                    st.markdown("🟢 **Test Used: One-Way ANOVA (3+ Groups)**", help="**The 'Big Picture' Test (for 3+ groups)**\n\n* **What it does:** It compares the averages (means) of three or more groups to see if at least one group is statistically different from the others.\n* **The Catch:** It tells you *that* a difference exists somewhere, but it does not tell you *which* specific groups are different.")
                    stat, p_val = stats.f_oneway(*data_series)
                    st.write(f"**F-statistic:** `{stat:.4f}` | **P-value:** `{p_val:.4e}`")
                    if p_val < 0.05:
                        perform_posthoc = True
                        posthoc_type = 'tukey'
                        
                else:
                    st.markdown("🟡 **Test Used: Kruskal-Wallis H Test**")
                    st.warning("Data Distribution Alert: Your data failed normality. Automatically switched to Kruskal-Wallis (the non-parametric equivalent of ANOVA for 3+ groups).")
                    stat, p_val = stats.kruskal(*data_series)
                    st.write(f"**H-statistic:** `{stat:.4f}` | **P-value:** `{p_val:.4e}`")
                    if p_val < 0.05:
                        perform_posthoc = True
                        posthoc_type = 'dunn'

                if p_val < 0.05:
                    st.success(f"**Interpretation:** There is a statistically significant difference in {anova_var} between the groups.")
                else:
                    st.info(f"**Interpretation:** There is NO statistically significant difference in {anova_var} between the groups.")

                # Step C: Post-Hoc Routing for 3+ Groups
                if perform_posthoc:
                    st.divider()
                    st.subheader("Pairwise Comparisons", help="**The 'Detective' Tests**\n\n* **What it does:** \"Post-hoc\" means \"after this.\" These are follow-up tests you run *only* if your ANOVA finds a significant difference. They do the detective work to find exactly where the differences lie.")
                    st.markdown("This post-hoc test identifies exactly which groupings are driving the significant difference found in the main test.")
                    
                    if posthoc_type == 'tukey':
                        st.markdown("🔵 **Post-Hoc Test: Tukey's HSD**", help="**A Specific Post-Hoc Test**\n\n* **What it does:** It systematically pairs up every single group against each other (A vs B, A vs C, B vs C) to tell you exactly which groups differ. It mathematically adjusts the rules so you don’t accidentally get false positives from running so many comparisons at once.")
                        flat_data = []
                        flat_labels = []
                        for g_name, g_data in groups_data.items():
                            flat_data.extend(g_data.tolist())
                            flat_labels.extend([g_name] * len(g_data))
                            
                        tukey = pairwise_tukeyhsd(endog=flat_data, groups=flat_labels, alpha=0.05)
                        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                        st.table(tukey_df)

                    elif posthoc_type == 'dunn':
                        st.markdown("🔵 **Post-Hoc Test: Dunn's Test**")
                        flat_data = []
                        flat_labels = []
                        for g_name, g_data in groups_data.items():
                            flat_data.extend(g_data.tolist())
                            flat_labels.extend([g_name] * len(g_data))
                        
                        df_posthoc = pd.DataFrame({'val': flat_data, 'group': flat_labels})
                        dunn_matrix = sp.posthoc_dunn(df_posthoc, val_col='val', group_col='group', p_adjust='bonferroni')
                        
                        pairs = []
                        labels_list = list(dunn_matrix.columns)
                        for r in range(len(labels_list)):
                            for c in range(r + 1, len(labels_list)):
                                g1 = labels_list[r]
                                g2 = labels_list[c]
                                p_adj = dunn_matrix.loc[g1, g2]
                                sig = p_adj < 0.05
                                pairs.append({
                                    "Comparison": f"{g1} vs. {g2}",
                                    "Adjusted P-Value": f"{p_adj:.4e}",
                                    "Significant (p<0.05)": "Yes" if sig else "No"
                                })
                                
                        dunn_df = pd.DataFrame(pairs)
                        st.table(dunn_df)
                
                st.subheader("Distribution Visualization")
                # Force Plotly to respect the numerical/logical category order rather than alphabetical
                cat_order = {plot_x: valid_groups}
                
                if is_normal:
                    fig = px.box(df_test, x=plot_x, y=anova_var, color=plot_x, points="all", title=f"Distribution of {anova_var} by {plot_x}", labels=LABEL_MAP, category_orders=cat_order)
                    fig.update_traces(boxmean=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.violin(df_test, x=plot_x, y=anova_var, color=plot_x, points="all", box=True, title=f"Distribution of {anova_var} by {plot_x}", labels=LABEL_MAP, category_orders=cat_order)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Showing a Violin & Swarm plot. Because this data does not follow a normal distribution, this chart highlights the data's density and Median, which are evaluated by the non-parametric test.")

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
                              labels=LABEL_MAP)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one GPU to visualize.")
        else:
            st.warning("No GPU names found in the datasets to track.")

    # --- 7. Q&A ---
    with tab7:
        st.header("7. Q&A")
        
        st.subheader("Category 1: Statistical Methodology & Rigor")
        st.markdown("The \"Why\" behind your tests is often where professors probe deepest.")
        
        with st.expander("Q1: You used a Mann-Whitney U test for \"NVIDIA Tax\" but a One-Way ANOVA for \"Wattage Analysis.\" Why change the test type?"):
            st.markdown("**The Armor:** You must explain your statistical workflow. The \"Historical Retail Price\" data was found to be heavily right-skewed, which strictly requires a non-parametric test like Mann-Whitney U to ensure accuracy. Conversely, the \"Watt Rating\" variable passed normality checks (Shapiro-Wilk), allowing for the more precise parametric One-Way ANOVA.")
            
        with st.expander("Q2: In your Mann-Whitney U test, you mention a U-statistic of 188. What does that number actually represent in this context?"):
            st.markdown("**The Armor:** With 24 cards per brand, there are 576 possible one-on-one price matchups. The U-statistic of 188 means AMD was the more expensive card in only 188 of those pairings, while NVIDIA dominated the remaining 388. This imbalance is what drove the significant $p = 0.040$ result.")
            
        with st.expander("Q3: Your Chi-Square test for grouped market tiers returned a $p$-value of exactly 1.000. Isn't a \"perfect\" result suspicious?"):
            st.markdown("**The Armor:** It is not an error but a result of flawless macro-architectural symmetry. Both manufacturers historically allocated exactly 13 units to the \"Budget Floor,\" 9 to the \"Enthusiast Core,\" and 2 to the \"Halo Tier\". The dataset is a perfectly mirrored matrix at the macro level, even if they differ at the micro level.")
            
        st.subheader("Category 2: Regression & Market Efficiency")
        st.markdown("These questions test if you understand the \"Economic Logic\" of your data.")
        
        with st.expander("Q4: Why does the secondary market ($R^2 = 0.8967$) correlate so much better with performance than the retail market ($R^2 = 0.68$)?"):
            st.markdown("**The Armor:** The retail market (MSRP) is heavily skewed by artificial product tiering, marketing hype, and brand premiums. In consumer-to-consumer sales (eBay), these factors are \"stripped away,\" distilling a GPU's financial value almost entirely down to its empirical rendering power (3DMark scores).")
            
        with st.expander("Q5: If 3DMark only explains 68% of retail price variance, what accounts for the remaining 32%?"):
            st.markdown("**The Armor:** Your research identifies these \"hidden\" drivers as brand culture, supply chain constraints, and the \"NVIDIA Tax\". The statistical findings prove that retail pricing is a \"moderately strong\" but ultimately noisy predictor of hardware capability compared to the used market.")
            
        st.subheader("Category 3: VRAM Plateaus & Strategic Manipulation")
        st.markdown("This is the \"Deep Insight\" section where you show you understand corporate strategy.")
        
        with st.expander("Q6: You claim 12GB and 16GB VRAM tiers are \"statistically indistinguishable\" ($p = 0.810$). Why does the market refuse to value that 4GB difference?"):
            st.markdown("**The Armor:** Tukey's HSD analysis shows that standard gamers (who dominate these tiers) do not view 16GB as a high-enough leap to command a premium over 12GB. The \"Memory Premium\" only triggers at the 24GB tier ($p = 0.002$), which is decoupled from gaming and driven by inelastic prosumer demand for AI and data science.")
            
        with st.expander("Q7: How do NVIDIA and AMD use \"Micro-Segmentation\" differently to manipulate consumers?"):
            st.markdown("**The Armor:** While they have identical \"bucket sizes\" (9 cards in the Enthusiast Core), their contents differ: AMD uses high-capacity 16GB models as a value differentiator. NVIDIA fragments their 9-card quota across 10GB, 11GB, and 12GB configurations to mathematically force aggressive upselling pathways.")
            
        st.subheader("Category 4: Architectural Efficiency")
        
        with st.expander("Q8: Your ANOVA showed no significant difference in power consumption ($p = 0.466$). What does this imply about NVIDIA's higher prices?"):
            st.markdown("**The Armor:** Since NVIDIA commands a verified retail premium but operates within an indistinguishable power envelope compared to AMD, it proves they aren't using \"brute force\" (more electricity) to win. Their premium is instead sustained by superior architectural efficiency or dominant brand perception.")
            
        st.subheader("Quick-Fire Statistical \"Armor\" (Cheat Sheet)")
        st.markdown("""
- **Sample Size:** 48 discrete GPUs.
- **Comparison of Averages:** NVIDIA: $612.49 retail vs. AMD: $387.53.
- **Wattage Noise:** The 20W difference (211W vs 191W) is statistically meaningless noise because the $F$-statistic was below 1.0.
- **Post-Hoc Test:** You used Tukey's HSD specifically to find which VRAM pairs were responsible for the overall variance.
""")

else:
    st.info("Awaiting data load to initialize analysis sections.")
