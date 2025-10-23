"""
Visualization utilities for creating charts and plots.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VisualizationHelper:
    """Helper class for creating visualizations."""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, 
                    chart_type: str = 'bar',
                    x_column: str = None,
                    y_column: str = None,
                    title: str = None,
                    color_column: str = None,
                    **kwargs) -> go.Figure:
        """
        Create a chart based on DataFrame and parameters.
        
        Args:
            df: DataFrame to visualize
            chart_type: Type of chart ('bar', 'line', 'scatter', 'pie', 'histogram')
            x_column: Column for x-axis
            y_column: Column for y-axis
            title: Chart title
            color_column: Column for color encoding
            **kwargs: Additional parameters
            
        Returns:
            Plotly figure object
        """
        try:
            # Auto-detect columns if not specified
            if x_column is None:
                x_column = df.columns[0]
            if y_column is None and len(df.columns) > 1:
                y_column = df.columns[1]
            
            # Create chart based on type
            if chart_type == 'bar':
                fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
            elif chart_type == 'line':
                fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
            elif chart_type == 'scatter':
                fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
            elif chart_type == 'pie':
                fig = px.pie(df, names=x_column, values=y_column, title=title)
            elif chart_type == 'histogram':
                fig = px.histogram(df, x=x_column, title=title)
            else:
                # Default to bar chart
                fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
            
            # Update layout
            fig.update_layout(
                title=title or f"{chart_type.title()} Chart",
                xaxis_title=x_column,
                yaxis_title=y_column or "Value",
                showlegend=True,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            return VisualizationHelper._create_error_chart(str(e))
    
    @staticmethod
    def create_trend_chart(df: pd.DataFrame,
                          time_column: str,
                          value_column: str,
                          group_column: str = None,
                          title: str = None) -> go.Figure:
        """
        Create a trend chart for time series data.
        
        Args:
            df: DataFrame with time series data
            time_column: Column containing time/date data
            value_column: Column containing values to plot
            group_column: Optional column for grouping
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Ensure time column is datetime
            df[time_column] = pd.to_datetime(df[time_column])
            
            if group_column:
                fig = px.line(df, x=time_column, y=value_column, color=group_column, title=title)
            else:
                fig = px.line(df, x=time_column, y=value_column, title=title)
            
            fig.update_layout(
                title=title or f"Trend Analysis: {value_column} over {time_column}",
                xaxis_title=time_column,
                yaxis_title=value_column,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trend chart: {str(e)}")
            return VisualizationHelper._create_error_chart(str(e))
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str = None) -> go.Figure:
        """
        Create a correlation heatmap for numeric columns.
        
        Args:
            df: DataFrame with numeric columns
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            
            if numeric_df.empty:
                return VisualizationHelper._create_error_chart("No numeric columns found")
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title=title or "Correlation Heatmap",
                height=500,
                xaxis_title="Variables",
                yaxis_title="Variables"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return VisualizationHelper._create_error_chart(str(e))
    
    @staticmethod
    def create_summary_dashboard(df: pd.DataFrame, title: str = None) -> go.Figure:
        """
        Create a summary dashboard with multiple charts.
        
        Args:
            df: DataFrame to summarize
            title: Dashboard title
            
        Returns:
            Plotly figure object with subplots
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Data Overview', 'Numeric Distribution', 'Missing Values', 'Data Types'),
                specs=[[{"type": "table"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Data overview table
            overview_data = {
                'Metric': ['Rows', 'Columns', 'Memory Usage', 'Numeric Columns', 'Text Columns'],
                'Value': [
                    len(df),
                    len(df.columns),
                    f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
                    len(df.select_dtypes(include=['number']).columns),
                    len(df.select_dtypes(include=['object']).columns)
                ]
            }
            
            fig.add_trace(
                go.Table(
                    header=dict(values=list(overview_data.keys())),
                    cells=dict(values=list(overview_data.values()))
                ),
                row=1, col=1
            )
            
            # Numeric distribution
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig.add_trace(
                    go.Histogram(x=df[numeric_cols[0]], name=numeric_cols[0]),
                    row=1, col=2
                )
            
            # Missing values
            missing_data = df.isnull().sum()
            fig.add_trace(
                go.Bar(x=missing_data.index, y=missing_data.values, name='Missing Values'),
                row=2, col=1
            )
            
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            fig.add_trace(
                go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, name='Data Types'),
                row=2, col=2
            )
            
            fig.update_layout(
                title=title or "Data Summary Dashboard",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {str(e)}")
            return VisualizationHelper._create_error_chart(str(e))
    
    @staticmethod
    def _create_error_chart(error_message: str) -> go.Figure:
        """Create an error chart when visualization fails."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Chart Error",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    @staticmethod
    def get_chart_recommendations(df: pd.DataFrame) -> List[str]:
        """
        Get chart type recommendations based on DataFrame characteristics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of recommended chart types
        """
        recommendations = []
        
        # Check data types
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        # Recommendations based on data characteristics
        if len(numeric_cols) >= 2:
            recommendations.append("scatter - Good for showing relationships between numeric variables")
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations.append("bar - Good for comparing categories")
        
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations.append("line - Good for showing trends over time")
        
        if len(categorical_cols) >= 1:
            recommendations.append("pie - Good for showing proportions")
        
        if len(numeric_cols) >= 1:
            recommendations.append("histogram - Good for showing distributions")
        
        if len(numeric_cols) >= 2:
            recommendations.append("correlation heatmap - Good for showing correlations")
        
        return recommendations


