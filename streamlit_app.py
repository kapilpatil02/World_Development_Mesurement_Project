"""
streamlit_app.py
================

Streamlit Web App for World Development Clustering Model

A simple, interactive web application to classify countries based on
their development indicators using a trained K-Means clustering model.

HOW TO RUN:
-----------
1. Make sure you've trained the model: python train_model.py
2. Install dependencies: pip install -r requirements.txt
3. Run the app: streamlit run streamlit_app.py
4. Open browser at: http://localhost:8501

"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from world_development_model import WorldDevelopmentClusteringModel
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="World Development Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stButton>button { width: 100%; background-color: #1f77b4; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CLUSTER INFORMATION
# ============================================================================

cluster_info = {
    0: {
        "name": "Developed Countries",
        "description": "High GDP, long life expectancy, high internet penetration, advanced economy",
        "color": "#2ecc71",
        "icon": "🌟"
    },
    1: {
        "name": "Developing Countries", 
        "description": "Lower GDP, shorter life expectancy, limited internet access, growing economy",
        "color": "#e74c3c",
        "icon": "📈"
    },
    2: {
        "name": "Emerging Economies",
        "description": "Middle-tier GDP, moderate development indicators, rapid growth potential",
        "color": "#f39c12",
        "icon": "🚀"
    }
}

# ============================================================================
# LOAD MODEL (CACHED FOR PERFORMANCE)
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    model_path = 'world_development_kmeans.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please run `python train_model.py` first to train and save the model.")
        st.stop()
    
    try:
        model = WorldDevelopmentClusteringModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load model
model = load_model()

# ============================================================================
# MAIN TITLE
# ============================================================================

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🌍")
with col2:
    st.markdown("# World Development Clustering Model")

st.markdown("### Classify countries into development clusters using Machine Learning")
st.divider()

# ============================================================================
# SIDEBAR - MODE SELECTION
# ============================================================================

st.sidebar.markdown("## 🎯 Navigation")
mode = st.sidebar.radio(
    "Select Mode:",
    ["Single Country", "Batch Upload", "View Dataset", "Insights & Analytics", "About"]
)

# Model info in sidebar
st.sidebar.divider()
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(f"""
**Model Type:** K-Means Clustering  
**Clusters:** 3  
**Silhouette Score:** ~0.22  
**Features Used:** 22 development indicators
""")

# ============================================================================
# MODE 1: SINGLE COUNTRY INPUT
# ============================================================================

if mode == "Single Country":
    st.header("📊 Enter Country Data for Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Basic Information")
        country_name = st.text_input("Country Name", "Example Country")
    
    with col2:
        st.subheader("🏦 Economic Indicators")
        gdp = st.number_input(
            "GDP (USD)", 
            min_value=0, 
            value=500000000000, 
            step=100000000,
            format="%d",
            help="Gross Domestic Product"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("❤️ Health Indicators")
        life_exp_female = st.slider("Life Expectancy Female (years)", 40.0, 90.0, 75.0)
        life_exp_male = st.slider("Life Expectancy Male (years)", 40.0, 90.0, 70.0)
        health_exp_capita = st.number_input(
            "Health Exp per Capita (USD)", 
            0, 20000, 2000,
            help="Healthcare spending per person per year"
        )
    
    with col2:
        st.subheader("💻 Technology")
        internet_usage = st.slider("Internet Usage (% of population)", 0.0, 100.0, 65.0, 1.0)
        mobile_usage = st.slider("Mobile Phone Usage (per 100 people)", 0.0, 200.0, 110.0, 5.0)
    
    with col3:
        st.subheader("👥 Demographics")
        population = st.number_input(
            "Population Total", 
            min_value=0, 
            value=50000000, 
            step=1000000,
            help="Total population in thousands"
        )
        birth_rate = st.slider("Birth Rate (per 1000)", 0.0, 50.0, 15.0, 1.0)
    
    # Additional features
    with st.expander("⚙️ Additional Indicators (Optional)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            co2 = st.number_input("CO2 Emissions (metric tons)", 0, 10000000000, 500000)
            business_tax = st.slider("Business Tax Rate (%)", 0.0, 50.0, 25.0, 1.0)
        with col2:
            energy = st.number_input("Energy Usage (kg oil equivalent)", 0, 10000, 2500)
            days_business = st.number_input("Days to Start Business", 0, 365, 7)
        with col3:
            tourism_in = st.number_input("Tourism Inbound (annual)", 0, 100000000, 10000000)
            tourism_out = st.number_input("Tourism Outbound (annual)", 0, 100000000, 8000000)
    
    # Predict button
    st.divider()
    if st.button("🔮 Predict Development Cluster", key="predict_single", use_container_width=True, type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Country': [country_name],
            'GDP': [gdp],
            'Population Total': [population],
            'Life Expectancy Female': [life_exp_female],
            'Life Expectancy Male': [life_exp_male],
            'Health Exp/Capita': [health_exp_capita],
            'Internet Usage': [internet_usage / 100],  # Convert to decimal
            'Mobile Phone Usage': [mobile_usage],
            'CO2 Emissions': [co2],
            'Energy Usage': [energy],
            'Tourism Inbound': [tourism_in],
            'Tourism Outbound': [tourism_out],
            'Business Tax Rate': [business_tax],
            'Days to Start Business': [days_business],
            'Birth Rate': [birth_rate / 1000],  # Convert to decimal
        })
        
        # Make prediction
        try:
            with st.spinner("🔄 Analyzing country data..."):
                cluster = model.predict(input_data)[0]
            
            cluster_data = cluster_info[cluster]
            
            # Display result
            st.success("✅ Classification Complete!")
            
            st.markdown(f"""
                <div style="background-color: {cluster_data['color']}20; 
                            padding: 2.5rem; 
                            border-radius: 10px; 
                            border-left: 5px solid {cluster_data['color']}">
                    <h2 style="color: {cluster_data['color']}; margin: 0;">
                        {cluster_data['icon']} {cluster_data['name']}
                    </h2>
                    <p style="font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;">
                        {cluster_data['description']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show input summary
            st.subheader("📋 Key Indicators")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("GDP", f"${gdp/1e12:.2f}T" if gdp >= 1e12 else f"${gdp/1e9:.0f}B")
            col2.metric("Life Expectancy (F)", f"{life_exp_female:.1f} yrs")
            col3.metric("Internet Usage", f"{internet_usage:.1f}%")
            col4.metric("Health Exp/Capita", f"${health_exp_capita:,.0f}")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please ensure all required fields are filled correctly.")

# ============================================================================
# MODE 2: BATCH FILE UPLOAD
# ============================================================================

elif mode == "Batch Upload":
    st.header("📁 Batch Classification - Upload Data File")
    
    st.info("📌 Upload an Excel (.xlsx) or CSV (.csv) file with country development data")
    
    uploaded_file = st.file_uploader(
        "Choose a file:",
        type=['xlsx', 'csv'],
        help="File should contain a 'Country' column and development indicators"
    )
    
    if uploaded_file is not None:
        try:
            # Load file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} records")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Predict button
            st.divider()
            if st.button("🔮 Classify All Countries", use_container_width=True, type="primary"):
                with st.spinner("🔄 Classifying countries..."):
                    try:
                        predictions = model.predict(df)
                        df['Cluster'] = predictions
                        df['Cluster_Name'] = df['Cluster'].map(lambda x: cluster_info[x]['name'])
                        
                        st.success("✅ Classification Complete!")
                        
                        # Cluster distribution
                        st.subheader("📊 Results Overview")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Cluster Distribution")
                            cluster_counts = df['Cluster_Name'].value_counts()
                            
                            fig = px.pie(
                                values=cluster_counts.values,
                                names=cluster_counts.index,
                                title="Countries per Cluster",
                                color_discrete_map={
                                    "Developed Countries": "#2ecc71",
                                    "Developing Countries": "#e74c3c",
                                    "Emerging Economies": "#f39c12"
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Summary Statistics")
                            for cluster_name in ['Developed Countries', 'Developing Countries', 'Emerging Economies']:
                                count = (df['Cluster_Name'] == cluster_name).sum()
                                percentage = (count / len(df)) * 100
                                cluster_icon = next(c['icon'] for c in cluster_info.values() if c['name'] == cluster_name)
                                st.metric(f"{cluster_icon} {cluster_name}", f"{count}", f"{percentage:.1f}%")
                        
                        # Show full results
                        st.divider()
                        st.subheader("📋 Classification Results")
                        
                        # Create display dataframe
                        result_df = df[['Country', 'Cluster_Name']].copy()
                        result_df = result_df.sort_values('Cluster_Name')
                        
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download Results (CSV)",
                            data=csv_data,
                            file_name='classified_countries.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Classification error: {str(e)}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Show example format
        st.info("💡 Need an example? Download a template below:")
        
        example_data = pd.DataFrame({
            'Country': ['Example Country A', 'Example Country B', 'Example Country C'],
            'GDP': [5000000000000, 1000000000000, 2500000000000],
            'Population Total': [50000000, 20000000, 35000000],
            'Life Expectancy Female': [80, 65, 72],
            'Life Expectancy Male': [75, 60, 68],
            'Health Exp/Capita': [3000, 500, 1200],
            'Internet Usage': [0.8, 0.3, 0.6],
            'Mobile Phone Usage': [130, 80, 110],
        })
        
        st.download_button(
            label="📥 Download Example Template",
            data=example_data.to_csv(index=False).encode('utf-8'),
            file_name='example_template.csv',
            mime='text/csv'
        )

# ============================================================================
# MODE 3: VIEW DATASET INFO
# ============================================================================

elif mode == "View Dataset":
    st.header("📊 Dataset Information")
    
    st.info("ℹ️ Information about the training dataset and cluster characteristics")
    
    st.subheader("🌍 Cluster Characteristics")
    
    for cluster_id, info in cluster_info.items():
        with st.expander(f"{info['icon']} {info['name']}", expanded=(cluster_id == 0)):
            st.markdown(f"**Description:** {info['description']}")
            st.divider()
            
            if cluster_id == 0:
                st.markdown("""
                **Typical Characteristics:**
                - High GDP per capita
                - Life expectancy > 75 years
                - Internet penetration > 70%
                - Advanced healthcare systems
                - Developed infrastructure
                """)
            elif cluster_id == 1:
                st.markdown("""
                **Typical Characteristics:**
                - Lower GDP per capita
                - Life expectancy < 65 years
                - Limited internet access (< 30%)
                - Basic healthcare systems
                - Developing infrastructure
                """)
            else:
                st.markdown("""
                **Typical Characteristics:**
                - Middle-tier GDP
                - Life expectancy 65-75 years
                - Internet penetration 30-70%
                - Improving healthcare
                - Developing infrastructure
                """)
    
    st.divider()
    st.subheader("📈 Feature Importance")
    st.markdown("""
    The model considers 22 development indicators:
    - **Economic:** GDP, Tax Rate, Tourism, Business indicators
    - **Health:** Life Expectancy, Health Spending, Mortality rates
    - **Technology:** Internet Usage, Mobile penetration
    - **Environment:** CO2 Emissions, Energy Usage
    - **Demographics:** Population, Birth rates, Urban percentage
    
    Features are standardized before clustering.
    """)

# ============================================================================
# MODE 4: INSIGHTS & ANALYTICS
# ============================================================================

elif mode == "Insights & Analytics":
    st.header("📈 Insights & Analytics")
    st.info("Explore dataset-level insights, correlations, PCA and cluster summaries.")

    # Load dataset
    try:
        with st.spinner("Loading dataset..."):
            df_ins = pd.read_excel('World_development_mesurement.xlsx')
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

    st.subheader("Dataset Overview")
    st.write(f"Records: {len(df_ins)} | Columns: {df_ins.shape[1]}")
    st.dataframe(df_ins.head(10), use_container_width=True)

    # Clean numeric-like object columns
    df_clean = df_ins.copy()
    obj_cols = df_clean.select_dtypes(include='object').columns
    for col in obj_cols:
        if col != 'Country':
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('%', '', regex=False)
                .str.strip()
            )
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Numeric frame for analysis
    num_df = df_clean.select_dtypes(include=['int64', 'float64']).copy()

    st.subheader("Top / Bottom Countries by Metric")
    metric = st.selectbox("Select metric:", options=num_df.columns.tolist(), index=0)
    if metric:
        top_n = st.slider("Top N / Bottom N", 1, 20, 5)
        top = df_clean[['Country', metric]].dropna().sort_values(metric, ascending=False).head(top_n)
        bot = df_clean[['Country', metric]].dropna().sort_values(metric, ascending=True).head(top_n)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Top {top_n} by {metric}**")
            st.table(top.set_index('Country'))
        with col2:
            st.markdown(f"**Bottom {top_n} by {metric}**")
            st.table(bot.set_index('Country'))

    st.subheader("Correlation Matrix")
    corr = num_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("PCA - Explained Variance")
    pca_cols = num_df.dropna(axis=1, how='all').columns
    if len(pca_cols) >= 2:
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(num_df[pca_cols])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=min(6, Xs.shape[1]))
        pca.fit(Xs)
        evr = pca.explained_variance_ratio_
        evr_df = pd.DataFrame({'PC': [f'PC{i+1}' for i in range(len(evr))], 'Explained Variance': evr})
        fig_pca = px.bar(evr_df, x='PC', y='Explained Variance', title='PCA Explained Variance Ratio')
        st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Cluster Summaries")
    try:
        with st.spinner("Computing cluster summaries..."):
            preds = model.predict(df_ins)

            # Use cleaned numeric frame for aggregation to avoid dtype issues
            df_num_for_clusters = num_df.copy()
            # Ensure alignment: if lengths mismatch, align by index
            if len(df_num_for_clusters) == len(df_ins):
                df_num_for_clusters['Cluster'] = preds
            else:
                # fallback: create new DataFrame combining Country and numeric cols
                temp = df_clean.select_dtypes(include=['int64', 'float64']).copy()
                temp['Cluster'] = preds
                df_num_for_clusters = temp

            # Map cluster names
            df_num_for_clusters['Cluster_Name'] = df_num_for_clusters['Cluster'].map(lambda x: cluster_info.get(x, {}).get('name', str(x)))

            # Distribution
            cluster_counts = df_num_for_clusters['Cluster_Name'].value_counts()
            fig = px.pie(values=cluster_counts.values, names=cluster_counts.index, title='Cluster Distribution',
                         color_discrete_map={"Developed Countries": "#2ecc71", "Developing Countries": "#e74c3c", "Emerging Economies": "#f39c12"})
            st.plotly_chart(fig, use_container_width=True)

            # Numeric summaries: mean, median, count
            display_cols = df_num_for_clusters.select_dtypes(include=['int64', 'float64']).columns.tolist()
            means = df_num_for_clusters.groupby('Cluster')[display_cols].mean()
            medians = df_num_for_clusters.groupby('Cluster')[display_cols].median()
            counts = df_num_for_clusters.groupby('Cluster').size().rename('Count')

            st.markdown("**Cluster counts**")
            st.table(counts)

            st.markdown("**Cluster feature means (numeric)**")
            means.index = [f'Cluster {i}' for i in means.index]
            st.dataframe(means.T, use_container_width=True)

            st.markdown("**Cluster feature medians (numeric)**")
            medians.index = [f'Cluster {i}' for i in medians.index]
            st.dataframe(medians.T, use_container_width=True)

            # PCA scatter (2D) colored by cluster for quick visual separation
            try:
                pca_cols_local = display_cols if len(display_cols) >= 2 else pca_cols
                imputer_local = SimpleImputer(strategy='median')
                Xp = imputer_local.fit_transform(df_num_for_clusters[pca_cols_local])
                scaler_local = StandardScaler()
                Xs_local = scaler_local.fit_transform(Xp)
                pca2 = PCA(n_components=2)
                pcs = pca2.fit_transform(Xs_local)
                pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = df_num_for_clusters['Cluster'].values
                # include country names if present
                if 'Country' in df_ins.columns and len(df_ins) == len(pca_df):
                    pca_df['Country'] = df_ins['Country'].values
                fig_scatter = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', hover_data=['Country'] if 'Country' in pca_df.columns else None, title='PCA (2 components) by Cluster')
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception:
                st.info("PCA scatter could not be generated for the current dataset.")

    except Exception as e:
        st.warning(f"Could not compute clusters: {e}")

    st.subheader("Outliers / Extreme Values")
    if not num_df.empty:
        zscores = (num_df - num_df.mean()) / num_df.std()
        thresh = st.slider('Z-score threshold', 2.0, 5.0, 3.0)
        outliers = (zscores.abs() > thresh)
        outlier_summary = outliers.sum().sort_values(ascending=False).head(10)
        st.table(outlier_summary)

    st.markdown("---")

    # Additional deep-insight panels
    st.subheader("Missing Values & Data Completeness")
    missing = df_clean.isnull().sum()
    missing_pct = (missing / len(df_clean)) * 100
    mv_df = pd.DataFrame({'missing': missing, 'missing_pct': missing_pct}).sort_values('missing_pct', ascending=False)
    if not mv_df.empty:
        st.dataframe(mv_df.head(20))
        fig_missing = px.bar(mv_df.reset_index().head(20), x='index', y='missing_pct', labels={'index':'Feature','missing_pct':'% missing'}, title='Top 20: Missing % by Feature')
        st.plotly_chart(fig_missing, use_container_width=True)

    st.subheader("Feature Distributions")
    dist_select = st.multiselect('Select features to plot distributions (histograms)', options=num_df.columns.tolist(), default=num_df.columns[:4].tolist())
    for feat in dist_select:
        try:
            fig_dist = px.histogram(df_clean, x=feat, nbins=40, title=f'Distribution: {feat}')
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception:
            st.info(f'Could not plot distribution for {feat}')

    st.subheader("Boxplot by Cluster")
    if 'Cluster' in df_num_for_clusters.columns:
        box_feat = st.selectbox('Feature for boxplot', options=[c for c in df_num_for_clusters.columns if c != 'Cluster'] , index=0)
        try:
            box_df = df_num_for_clusters.copy()
            if 'Country' in df_ins.columns and len(df_ins) == len(box_df):
                box_df['Country'] = df_ins['Country'].values
            fig_box = px.box(box_df, x='Cluster', y=box_feat, points='outliers', title=f'Boxplot of {box_feat} by Cluster')
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.info(f'Boxplot error: {e}')

    st.subheader('Feature separation (between-cluster / total variance)')
    try:
        total_var = num_df.var()
        between_var = means.var()
        # Align indexes
        between_var = between_var.reindex(total_var.index).fillna(0)
        separation = (between_var / total_var).sort_values(ascending=False).head(20)
        sep_df = pd.DataFrame({'separation_ratio': separation})
        st.table(sep_df)
        fig_sep = px.bar(sep_df.reset_index(), x='index', y='separation_ratio', labels={'index':'Feature','separation_ratio':'Between/Total Var'}, title='Top features by cluster separation')
        st.plotly_chart(fig_sep, use_container_width=True)
    except Exception:
        st.info('Could not compute feature separation metric')

    st.subheader('Correlation Explorer')
    corr_feature = st.selectbox('Select feature to show top correlations for', options=num_df.columns.tolist(), index=0)
    if corr_feature:
        corr_series = corr[corr_feature].drop(labels=[corr_feature])
        top_corr = corr_series.abs().sort_values(ascending=False).head(10)
        st.table(pd.DataFrame({'feature': top_corr.index, 'corr': corr_series[top_corr.index].values}))

    st.subheader('Silhouette Score')
    try:
        if len(set(preds)) > 1:
            imputer_global = SimpleImputer(strategy='median')
            X_full = imputer_global.fit_transform(num_df.dropna(axis=1, how='all'))
            scaler_global = StandardScaler()
            Xs_full = scaler_global.fit_transform(X_full)
            sil = silhouette_score(Xs_full, preds)
            st.metric('Silhouette Score', f'{sil:.4f}')
        else:
            st.info('Silhouette not available — only one cluster present')
    except Exception:
        st.info('Silhouette could not be computed for the current dataset')

    # End Insights section

    # ============================================================================
    # MODE 4: ABOUT
    # ============================================================================

elif mode == "About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    Classify countries based on development indicators using K-Means clustering.
    Countries are grouped into: Developed, Developing, and Emerging Economies.
    
    ### 📊 Model Details
    - **Algorithm:** K-Means Clustering
    - **Clusters:** 3
    - **Features:** 22 development indicators
    - **Preprocessing:** Median imputation, log transformation, standardization
    
    ### 🔧 Technologies
    - Python 3.8+ | Scikit-Learn | Streamlit | Pandas | Plotly
    
    ### 🚀 Usage
    **Train the model:**
    ```bash
    python train_model.py
    ```
    
    **Run the web app:**
    ```bash
    streamlit run streamlit_app.py
    ```
    
    ### ⚠️ Disclaimer
    This model is for informational purposes. Always consult domain experts
    for policy or investment decisions.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style="text-align: center; color: #999;">
        <p style="margin: 0;">
            🤖 K-Means Clustering | 🌍 World Development Indicators | Built with ❤️ using Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)
