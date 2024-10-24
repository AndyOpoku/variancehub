import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
    
    def ask(self, query):
        """Simple natural language query processor"""
        query = query.lower()
        try:
            if 'average' in query or 'mean' in query:
                col = self._extract_column_name(query)
                if col:
                    return f"Average of {col}: {self.df[col].mean():.2f}"
                return "Column-wise means:\n" + str(self.df.mean())
            
            elif 'missing' in query:
                missing_counts = self.df.isnull().sum()
                return f"Missing values per column:\n{missing_counts}"
            
            elif 'correlation' in query:
                return "Correlation matrix:\n" + str(self.df.corr().round(2))
            
            elif 'unique' in query:
                col = self._extract_column_name(query)
                if col:
                    return f"Unique values in {col}:\n{self.df[col].nunique()}"
                return "Please specify a column name"
            
            elif 'maximum' in query or 'max' in query:
                col = self._extract_column_name(query)
                if col:
                    return f"Maximum of {col}: {self.df[col].max()}"
                return "Column-wise maximums:\n" + str(self.df.max())
            
            elif 'minimum' in query or 'min' in query:
                col = self._extract_column_name(query)
                if col:
                    return f"Minimum of {col}: {self.df[col].min()}"
                return "Column-wise minimums:\n" + str(self.df.min())
            
            else:
                return "I don't understand the question. Try asking about averages, missing values, correlations, unique values, maximums, or minimums."
        
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def apply(self, query):
        """Apply operations to the dataframe"""
        query = query.lower()
        try:
            if 'mean' in query:
                return self.df.mean()
            
            elif 'outliers' in query:
                col = self._extract_column_name(query)
                if col and col in self.df.select_dtypes(include=[np.number]).columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))][col]
                    return pd.DataFrame({'Outliers': outliers})
                return "Please specify a numeric column name"
            
            elif 'group by' in query:
                group_col = query.split('group by')[1].split('and')[0].strip()
                if 'mean' in query:
                    return self.df.groupby(group_col).mean()
                elif 'sum' in query:
                    return self.df.groupby(group_col).sum()
                return self.df.groupby(group_col).describe()
            
            else:
                return "Operation not recognized. Try 'calculate mean', 'find outliers', or 'group by'"
        
        except Exception as e:
            return f"Error applying operation: {str(e)}"
    
    def how(self, query):
        """Provide explanations for common data analysis tasks"""
        query = query.lower()
        
        explanations = {
            'missing values': """To handle missing values:
1. Check missing values: df.isnull().sum()
2. Options:
   - Drop missing values: df.dropna()
   - Fill with mean: df.fillna(df.mean())
   - Fill with median: df.fillna(df.median())
   - Fill with mode: df.fillna(df.mode())""",
            
            'normalize': """To normalize data:
1. Min-Max scaling: (x - min) / (max - min)
2. Z-score: (x - mean) / std
3. Using sklearn:
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   normalized_data = scaler.fit_transform(data)""",
            
            'correlations': """To find correlations:
1. Pearson correlation: df.corr()
2. Spearman correlation: df.corr(method='spearman')
3. Visualize: sns.heatmap(df.corr())""",
        }
        
        for key, explanation in explanations.items():
            if key in query:
                return explanation
        
        return "Topic not found. Try asking about 'missing values', 'normalize', or 'correlations'"
    
    def _extract_column_name(self, query):
        """Helper function to extract column names from query"""
        for col in self.df.columns:
            if col.lower() in query:
                return col
        return None

st.set_page_config(page_title="Data Visualization", layout="centered")
st.title("Data Visualization")
st.sidebar.title("Navigation")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # Initialize DataAnalyzer
        analyzer = DataAnalyzer(df)

        # Add Data Analysis Section
        st.write("### Data Analysis")
        
        # Ask Query
        st.subheader("Ask Questions About Your Data")
        ask_query = st.text_input("Ask a question about your data:", 
                                 placeholder="e.g., What is the average of column X?")
        if ask_query:
            try:
                ask_result = analyzer.ask(ask_query)
                st.write("Answer:", ask_result)
            except Exception as e:
                st.error(f"Error in ask query: {str(e)}")

        # Apply Query
        st.subheader("Apply Operations to Your Data")
        apply_query = st.text_input("Enter an operation to apply:", 
                                   placeholder="e.g., calculate mean of each column")
        if apply_query:
            try:
                apply_result = analyzer.apply(apply_query)
                st.write("Result:")
                st.write(apply_result)
            except Exception as e:
                st.error(f"Error in apply query: {str(e)}")

        # How Query
        st.subheader("Get Explanations About Your Data")
        how_query = st.text_input("Ask how to analyze something:", 
                                 placeholder="e.g., How to handle missing values?")
        if how_query:
            try:
                how_result = analyzer.how(how_query)
                st.write("Explanation:", how_result)
            except Exception as e:
                st.error(f"Error in how query: {str(e)}")

        
        # Data Preview
        st.write("### Data Preview")
        st.dataframe(df)
        
        # Basic Statistics
        st.write("### Basic Statistics")
        st.write(df.describe())
        st.subheader("First Five Rows")
        st.write(df.head())
        st.subheader("Last Five Rows")
        st.write(df.tail())
        st.subheader("Number of Rows and Columns")
        st.write(df.shape)
        st.subheader("Number of Missing Values")
        st.write(df.isnull().sum())

        # Only show numeric columns for certain plots
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_columns = df.columns.tolist()

        # Scatter Plot

        # Create scatter plot subsection
        st.subheader("Scatter Plot")

        # Add style customization options in sidebar
        color_palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
        grid_styles = ['whitegrid', 'darkgrid', 'white', 'dark']
        marker_styles = ['o', 's', '^', 'D', 'v', 'p']  # circle, square, triangle up, diamond, triangle down, pentagon

        # Style selection dropdowns
        selected_palette = st.sidebar.selectbox("Color Palette for Scatter Plot", color_palettes, key='scatter_palette')
        selected_grid = st.sidebar.selectbox("Grid Style for Scatter Plot", grid_styles, key='scatter_grid')
        selected_marker = st.sidebar.selectbox("Marker Style for Scatter Plot", marker_styles, key='scatter_marker')

        # Point size slider
        point_size = st.sidebar.slider("Point Size for Scatter Plot", min_value=20, max_value=200, value=80, key='scatter_size')

        # Select columns for scatter plot
        x_axis = st.sidebar.selectbox("X-axis for scatter plot", numeric_columns, key='scatter_x')
        y_axis = st.sidebar.selectbox("Y-axis for scatter plot", numeric_columns, key='scatter_y')

        # Set the style and color palette
        sns.set_style(selected_grid)
        sns.set_palette(selected_palette)

        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create scatter plot using seaborn
        sns.scatterplot(
            data=df,
            x=x_axis,
            y=y_axis,
            ax=ax,
            s=point_size,  # Point size
            marker=selected_marker,  # Marker style
            alpha=0.6,  # Some transparency for overlapping points
            edgecolor='white'  # White edge for better visibility
        )

        # Customize the plot
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'Scatter Plot of {x_axis} vs {y_axis}')

        # Add a subtle grid if not already present in style
        if selected_grid in ['white', 'dark']:
            ax.grid(True, linestyle='--', alpha=0.6)

        # Add correlation coefficient in the corner
        correlation = df[x_axis].corr(df[y_axis])
        ax.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Reset the style for other plots
        sns.reset_defaults()

        # Clear the plot from memory
        plt.close()

        
        # Pie Chart

        st.subheader("Pie Chart")
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            if not categorical_columns:
                st.write("No categorical columns available for pie chart")
            else:
                pie_column = st.sidebar.selectbox("Column for Pie Chart", categorical_columns, key='pie')
                
                if pie_column is not None:
                    # Calculate value counts for the selected column
                    pie_data = df[pie_column].value_counts()
                    
                    if not pie_data.empty:
                        # Create interactive pie chart using plotly
                        fig = go.Figure(data=[go.Pie(
                            labels=pie_data.index,
                            values=pie_data.values,
                            hole=0.3,  # Makes it a donut chart, remove or set to 0 for traditional pie
                            hovertemplate="<b>%{label}</b><br>" +
                                        "Count: %{value}<br>" +
                                        "Percentage: %{percent}<extra></extra>"
                        )])

                        # Update layout
                        fig.update_layout(
                            title=dict(
                            text=f'Distribution of {pie_column}',
                            automargin=True,
                            x=0.5,
                            xanchor='center',
                            y=0.97,
                            font=dict(size=25)
                    
                    ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.5,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=15)
                    ))

                        # Display the chart in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write(f"No data to plot for column '{pie_column}'")
                else:
                    st.write("Please select a column for the pie chart")

        except Exception as e:
            st.error(f"An error occurred while creating the pie chart: {str(e)}")
                
            ax.axis('equal')
            
            # Add title with seaborn style
            plt.title(f'Distribution of {pie_column}', pad=20, fontsize=12)
                
            st.pyplot(fig)
            plt.close()

        # Line Chart

        # Create line chart subsection
        st.subheader("Line Chart")

        # Add style customization options in sidebar
        color_palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
        grid_styles = ['whitegrid', 'darkgrid', 'white', 'dark']

        selected_palette = st.sidebar.selectbox("Color Palette for Line Chart", color_palettes, key='palette')
        selected_grid = st.sidebar.selectbox("Grid Style for Line Chart", grid_styles, key='grid')

        # Select columns for x and y axes from sidebar
        line_x_axis = st.sidebar.selectbox("X-Axis for Line Chart", all_columns, key='line_x')
        line_y_axis = st.sidebar.selectbox("Y-Axis for Line Chart", numeric_columns, key='line_y')

        # Set the style and color palette
        sns.set_style(selected_grid)
        sns.set_palette(selected_palette)

        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create line chart using seaborn
        sns.lineplot(
            data=df,
            x=line_x_axis,
            y=line_y_axis,
            ax=ax,
            marker='o',  # Add markers at data points
            markeredgecolor='white',  # Add white edge to markers for better visibility
            markeredgewidth=1
        )

        # Customize the plot
        plt.xlabel(line_x_axis)
        plt.ylabel(line_y_axis)
        plt.title(f'Line Chart of {line_x_axis} vs {line_y_axis}')
        plt.xticks(rotation=45)

        # Add a subtle grid if not already present in style
        if selected_grid in ['white', 'dark']:
            ax.grid(True, linestyle='--', alpha=0.6)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Reset the style for other plots
        sns.reset_defaults()

        # Clear the plot from memory
        plt.close()

        # Histogram
        
        # Create histogram subsection
        st.subheader("Histogram")

        # Select column for histogram from sidebar
        hist_column = st.sidebar.selectbox("Column for Histogram", numeric_columns, key='hist')

        # Create histogram using plotly express
        fig = px.histogram(
            df,
            x=hist_column,
            nbins=30,
            title=f'Histogram of {hist_column}',
            labels={hist_column: hist_column, 'count': 'Frequency'}
        )

        # Update layout for better appearance
        fig.update_layout(
            showlegend=False,
            xaxis_title=hist_column,
            yaxis_title='Frequency',
            bargap=0.1
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap

        # Create correlation heatmap subsection
        st.subheader("Correlation Heatmap")

        # Add style customization options in sidebar
        color_maps = ['coolwarm', 'viridis', 'magma', 'RdYlBu', 'RdBu_r', 'vlag']
        selected_cmap = st.sidebar.selectbox("Color Scheme for Heatmap", color_maps, key='heatmap_cmap')

        # Font size slider
        font_size = st.sidebar.slider("Annotation Font Size", 6, 12, 8, key='heatmap_font')

        # Mask option for upper/lower triangle
        mask_options = ['None', 'Upper Triangle', 'Lower Triangle']
        mask_choice = st.sidebar.selectbox("Show/Hide Triangle", mask_options, key='heatmap_mask')

        if len(numeric_columns) > 1:
            # Calculate correlation matrix
            correlation_matrix = df[numeric_columns].corr()
            
            # Create mask based on selection
            mask = np.zeros_like(correlation_matrix)
            if mask_choice == 'Upper Triangle':
                mask[np.triu_indices_from(mask, k=1)] = True
            elif mask_choice == 'Lower Triangle':
                mask[np.tril_indices_from(mask, k=-1)] = True
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create custom diverging palette
            if selected_cmap in ['coolwarm', 'RdYlBu', 'RdBu_r', 'vlag']:
                vmin, vmax = -1, 1
            else:
                vmin, vmax = correlation_matrix.min().min(), 1
            
            # Create heatmap with enhanced styling
            sns.heatmap(correlation_matrix,
                        annot=True,
                        cmap=selected_cmap,
                        center=0,
                        vmin=vmin,
                        vmax=vmax,
                        fmt='.2f',
                        square=True,
                        mask=mask if mask_choice != 'None' else None,
                        cbar_kws={'shrink': .8,
                                'label': 'Correlation Coefficient'},
                        annot_kws={'size': font_size},
                        ax=ax)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add title with custom styling
            plt.title('Correlation Heatmap', 
                    pad=20, 
                    size=14, 
                    fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            
            # Display the plot in Streamlit
            st.pyplot(fig)
            
            # Add correlation strength summary
            st.markdown("### Correlation Strength Summary")
            
            # Create columns for strong correlations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strong Positive Correlations (> 0.7)**")
                strong_pos = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        if correlation_matrix.iloc[i,j] > 0.7:
                            strong_pos.append(f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i,j]:.2f}")
                if strong_pos:
                    for corr in strong_pos:
                        st.write(corr)
                else:
                    st.write("None found")
            
            with col2:
                st.markdown("**Strong Negative Correlations (< -0.7)**")
                strong_neg = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        if correlation_matrix.iloc[i,j] < -0.7:
                            strong_neg.append(f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i,j]:.2f}")
                if strong_neg:
                    for corr in strong_neg:
                        st.write(corr)
                else:
                    st.write("None found")
            
            # Clear the plot from memory
            plt.close()
        else:
            st.warning("Need at least 2 numeric columns to create a correlation heatmap.")

    except:
        st.write("Please upload an Excel file to proceed")

# Add example queries in the sidebar
with st.sidebar.expander("Example Queries"):
    st.markdown("""
    ### Ask Examples:
    - What is the average of [column_name]?
    - How many missing values are there?
    - What is the correlation between columns?
    - What is the maximum of [column_name]?
    - What is the minimum of [column_name]?

    ### Apply Examples:
    - calculate mean of each column
    - find outliers in [column_name]
    - group by [column] and calculate mean

    ### How Examples:
    - How to handle missing values?
    - How to normalize the data?
    - How to find correlations?
    """)
