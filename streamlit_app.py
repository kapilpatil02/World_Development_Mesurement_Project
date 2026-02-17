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
        "color": "#2ecc71"
    },
    1: {
        "name": "Developing Countries", 
        "description": "Lower GDP, shorter life expectancy, limited internet access, growing economy",
        "color": "#e74c3c"
    },
    2: {
        "name": "Emerging Economies",
        "description": "Middle-tier GDP, moderate development indicators, rapid growth potential",
        "color": "#f39c12"
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
        st.error(f"Model file not found: {model_path}")
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
    st.markdown("# ")
with col2:
    st.markdown("# World Development Clustering Model")

st.markdown("### Classify countries into development clusters using Machine Learning")
st.divider()

# ============================================================================
# SIDEBAR - MODE SELECTION
# ============================================================================

st.sidebar.markdown("## Navigation")
mode = st.sidebar.radio(
    "Select Mode:",
    ["Batch Upload", "Insights & Analytics"]
)

# Model info in sidebar
st.sidebar.divider()
st.sidebar.markdown("### Model Info")
st.sidebar.info(f"""
**Model Type:** K-Means Clustering  
**Clusters:** 3  
**Silhouette Score:** ~0.22  
**Features Used:** 22 development indicators
""")



# ============================================================================
# MODE 2: BATCH FILE UPLOAD
# ============================================================================

if mode == "Batch Upload":
    st.header("Batch Classification - Upload Data File")
    
    st.info("Upload an Excel (.xlsx) or CSV (.csv) file with country development data")
    
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
            
            st.success(f"File loaded successfully! Found {len(df)} records")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Predict button
            st.divider()
            if st.button("Classify All Countries", use_container_width=True, type="primary"):
                with st.spinner("Classifying countries..."):
                    try:
                        predictions = model.predict(df)
                        df['Cluster'] = predictions
                        df['Cluster_Name'] = df['Cluster'].map(lambda x: cluster_info[x]['name'])
                        
                        st.success("Classification Complete!")
                        
                        # Cluster distribution
                        st.subheader("Results Overview")
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
                                st.metric(f"{cluster_name}", f"{count}", f"{percentage:.1f}%")
                        
                        # Show full results
                        st.divider()
                        st.subheader("Classification Results")
                        
                        # Create display dataframe
                        result_df = df[['Country', 'Cluster_Name']].copy()
                        result_df = result_df.sort_values('Cluster_Name')
                        
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv_data,
                            file_name='classified_countries.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Classification error: {str(e)}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Show example format
        st.info("Need an example? Download a template below:")
        
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
            label="Download Example Template",
            data=example_data.to_csv(index=False).encode('utf-8'),
            file_name='example_template.csv',
            mime='text/csv'
        )

# ============================================================================
# MODE 4: INSIGHTS & ANALYTICS
# ============================================================================

elif mode == "Insights & Analytics":
    st.header("Insights & Analytics")
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
    preds = None
    df_num_for_clusters = pd.DataFrame()
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
    if not df_num_for_clusters.empty and 'Cluster' in df_num_for_clusters.columns:
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
        if preds is not None and len(set(preds)) > 1:
            imputer_global = SimpleImputer(strategy='median')
            X_full = imputer_global.fit_transform(num_df.dropna(axis=1, how='all'))
            scaler_global = StandardScaler()
            Xs_full = scaler_global.fit_transform(X_full)
            sil = silhouette_score(Xs_full, preds)
            st.metric('Silhouette Score', f'{sil:.4f}')
        else:
            st.info('Silhouette not available — only one cluster present or predictions unavailable')
    except Exception:
        st.info('Silhouette could not be computed for the current dataset')

    # End Insights section

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style="text-align: center; color: #999;">
        <p style="margin: 0;">
            K-Means Clustering |  World Development Indicators
        </p>
    </div>
""", unsafe_allow_html=True)
