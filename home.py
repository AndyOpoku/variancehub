import streamlit as st

st.set_page_config(page_title="Variance Inc. Hub", layout="centered")

st.title("Welcome to Variance Inc. Playground")

st.markdown("""
### 🎯 About This App
This is a simple interactive data visualization tool that helps you to explore and understand your data through simple queries and visual representations.

### ✨ Key Features:
- 📊 Multiple visualization types
- 📈 Interactive plots
- 📉 Statistical analysis
- 🔍 Data exploration capabilities

### 🚀 Getting Started:
1. Navigate to the 'Visualization' page using the sidebar
2. Upload your CSV or Excel file
3. Explore different visualization options
4. Ask simple questions about your data

### 📝 Supported File Types:
- CSV (.csv)
- Excel (.xlsx)

### 📊 Available Visualizations:
- Scatter Plots
- Pie Charts
- Line Charts
- Histograms
- Heatmaps

### 💡 Tips:
- Make sure your data is clean and properly formatted
- Numeric columns will be used for scatter plots and histograms
- Categorical columns will be used for pie charts

### 📊 Taking it Further with Us:
- 📊 Machine Learning Model using your Dataset
- 📈 Custom Dashboard for your Data (E.D.A Web Applications)
- 📉 Custom Chatbot with A.I Capabilities
- 🔍 Data Entry and Analysis

### 🎯 Contact Us:
Email: varianceincglobal@gmail.com | call/whatsapp: +233 (0) 504 268 179 | facebook & twitter: Variance Inc.
""")

st.sidebar.success("Select a page above.")