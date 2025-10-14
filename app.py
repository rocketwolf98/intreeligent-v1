import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Tree Cluster Analysis Pipeline", layout="wide")

# Title
st.title("🌳 Tree Cluster Analysis Pipeline")

# Sidebar or Main Layout
col1, col2 = st.columns([2, 1])  # wider left column

with col1:
    st.subheader("Input Data")
    uploaded_file = st.file_uploader("Orthographic Map", type=["png", "jpg", "tif"])
    
    st.subheader("Processing Parameters")
    grid_size = st.number_input("Grid Size (px)", value=256, min_value=1)
    overlap = st.number_input("Overlap (%)", value=10, min_value=0, max_value=100)
    cluster_count = st.number_input("Cluster Count", value=5, min_value=1)
    min_tree_area = st.number_input("Min Tree Area", value=50, min_value=1)

    process_clicked = st.button("🚀 Start Processing")

    if uploaded_file is not None and process_clicked:
        st.success("Processing complete! ✅")
        
        # Show uploaded image
        st.image(uploaded_file, caption="Uploaded Orthographic Map", use_column_width=True)
        
        # Simulated output: a scatter plot for "clusters"
        x = np.random.rand(50)
        y = np.random.rand(50)
        labels = np.random.randint(1, cluster_count + 1, size=50)

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=labels, cmap="tab10")
        ax.set_title("Clustered Tree Distribution (Simulated)")
        st.pyplot(fig)

    elif uploaded_file is None:
        st.info("⬆ Please upload a map to begin analysis")

with col2:
    st.subheader("Pipeline Status")
    st.checkbox("1. Grid Segmentation", value=process_clicked)
    st.checkbox("2. Autoencoder Inference", value=process_clicked)
    st.checkbox("3. Cluster Analysis", value=process_clicked)
    st.checkbox("4. Color Mapping", value=process_clicked)

    st.subheader("Statistics")
    if uploaded_file is not None and process_clicked:
        st.metric("Total Trees", np.random.randint(80, 150))
        st.metric("Distinct Clusters", cluster_count)
        st.metric("Grid Cells", np.random.randint(10, 30))
    else:
        st.metric("Total Trees", "--")
        st.metric("Distinct Clusters", "--")
        st.metric("Grid Cells", "--")

    st.subheader("Cluster Distribution")
    if uploaded_file is not None and process_clicked:
        df = pd.DataFrame({
            "Cluster": [f"Cluster {i}" for i in range(1, cluster_count+1)],
            "Trees": np.random.randint(10, 40, size=cluster_count)
        })
        st.bar_chart(df.set_index("Cluster"))
    else:
        st.write("📊 No data yet")