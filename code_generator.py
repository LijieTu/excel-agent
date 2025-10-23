"""
Python code generation system for Excel analysis based on natural language queries.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from nlp_agent import AnalysisIntent, AnalysisType
from config import Config

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Generates Python code for Excel analysis based on analysis intent."""
    
    def __init__(self):
        self.generated_code_history = []
        self.code_templates = self._load_code_templates()
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code templates for different analysis types."""
        return {
            "imports": """
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
""",
            "data_loading": """
# Load and prepare data
df = pd.read_excel('{file_path}', sheet_name='{sheet_name}')
df = df.dropna(how='all').dropna(axis=1, how='all')
df = df.reset_index(drop=True)
""",
            "sum": """
# Sum analysis
result = df.groupby({group_by})['{target_column}'].sum().reset_index()
result = result.sort_values('{target_column}', ascending={ascending})
""",
            "average": """
# Average analysis
result = df.groupby({group_by})['{target_column}'].mean().reset_index()
result = result.sort_values('{target_column}', ascending={ascending})
""",
            "count": """
# Count analysis
result = df.groupby({group_by}).size().reset_index(name='count')
result = result.sort_values('count', ascending={ascending})
""",
            "trend": """
# Trend analysis over time
df['{time_column}'] = pd.to_datetime(df['{time_column}'])
result = df.groupby([pd.Grouper(key='{time_column}', freq='{freq}')] + {group_by})['{target_column}'].sum().reset_index()
result = result.sort_values('{time_column}')
""",
            "correlation": """
# Correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
result = correlation_matrix
""",
            "filter": """
# Filter data
{filter_conditions}
result = df
""",
            "sort": """
# Sort data
result = df.sort_values({sort_columns}, ascending={ascending})
""",
            "visualization": """
# Create visualization
fig = px.{chart_type}(result, {chart_params})
fig.update_layout(
    title='{title}',
    xaxis_title='{x_title}',
    yaxis_title='{y_title}'
)
fig.show()
"""
        }
    
    def generate_code(self, intent: AnalysisIntent, file_path: str, sheet_name: str = None) -> str:
        """
        Generate Python code based on analysis intent.
        
        Args:
            intent: AnalysisIntent object containing analysis requirements
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to analyze
            
        Returns:
            Generated Python code as string
        """
        try:
            code_parts = []
            
            # Add imports
            code_parts.append(self.code_templates["imports"])
            
            # Add data loading
            data_loading_code = self.code_templates["data_loading"].format(
                file_path=file_path,
                sheet_name=sheet_name or "Sheet1"
            )
            code_parts.append(data_loading_code)
            
            # Add data preview
            code_parts.append(self._generate_data_preview_code())
            
            # Generate analysis code based on intent
            analysis_code = self._generate_analysis_code(intent)
            code_parts.append(analysis_code)
            
            # Add visualization if requested
            if intent.chart_type:
                viz_code = self._generate_visualization_code(intent)
                code_parts.append(viz_code)
            
            # Add result display
            code_parts.append(self._generate_result_display_code(intent))
            
            # Combine all code parts
            full_code = "\n".join(code_parts)
            
            # Store in history
            self.generated_code_history.append({
                "intent": intent,
                "code": full_code,
                "timestamp": pd.Timestamp.now()
            })
            
            return full_code
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return self._generate_error_code(str(e))
    
    def _generate_analysis_code(self, intent: AnalysisIntent) -> str:
        """Generate analysis code based on intent type."""
        if intent.analysis_type == AnalysisType.SUM:
            return self._generate_sum_code(intent)
        elif intent.analysis_type == AnalysisType.AVERAGE:
            return self._generate_average_code(intent)
        elif intent.analysis_type == AnalysisType.COUNT:
            return self._generate_count_code(intent)
        elif intent.analysis_type == AnalysisType.TREND:
            return self._generate_trend_code(intent)
        elif intent.analysis_type == AnalysisType.CORRELATION:
            return self._generate_correlation_code(intent)
        elif intent.analysis_type == AnalysisType.FILTER:
            return self._generate_filter_code(intent)
        elif intent.analysis_type == AnalysisType.SORT:
            return self._generate_sort_code(intent)
        elif intent.analysis_type == AnalysisType.GROUP_BY:
            return self._generate_group_by_code(intent)
        else:
            return self._generate_basic_analysis_code(intent)
    
    def _generate_sum_code(self, intent: AnalysisIntent) -> str:
        """Generate code for sum analysis."""
        group_by = self._format_group_by(intent.group_by_columns)
        target_col = intent.target_columns[0] if intent.target_columns else "value"
        
        code = f"""
# Sum analysis
result = df.groupby({group_by})['{target_col}'].sum().reset_index()
result = result.sort_values('{target_col}', ascending={str(intent.sort_ascending)})
"""
        return code
    
    def _generate_average_code(self, intent: AnalysisIntent) -> str:
        """Generate code for average analysis."""
        group_by = self._format_group_by(intent.group_by_columns)
        target_col = intent.target_columns[0] if intent.target_columns else "value"
        
        code = f"""
# Average analysis
result = df.groupby({group_by})['{target_col}'].mean().reset_index()
result = result.sort_values('{target_col}', ascending={str(intent.sort_ascending)})
"""
        return code
    
    def _generate_count_code(self, intent: AnalysisIntent) -> str:
        """Generate code for count analysis."""
        group_by = self._format_group_by(intent.group_by_columns)
        
        code = f"""
# Count analysis
result = df.groupby({group_by}).size().reset_index(name='count')
result = result.sort_values('count', ascending={str(intent.sort_ascending)})
"""
        return code
    
    def _generate_trend_code(self, intent: AnalysisIntent) -> str:
        """Generate code for trend analysis."""
        time_col = intent.time_column or "date"
        group_by = self._format_group_by(intent.group_by_columns)
        target_col = intent.target_columns[0] if intent.target_columns else "value"
        
        code = f"""
# Trend analysis over time
df['{time_col}'] = pd.to_datetime(df['{time_col}'])
result = df.groupby([pd.Grouper(key='{time_col}', freq='M')] + {group_by})['{target_col}'].sum().reset_index()
result = result.sort_values('{time_col}')
"""
        return code
    
    def _generate_correlation_code(self, intent: AnalysisIntent) -> str:
        """Generate code for correlation analysis."""
        code = """
# Correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    correlation_matrix = df[numeric_cols].corr()
    result = correlation_matrix
else:
    result = df.describe()
"""
        return code
    
    def _generate_filter_code(self, intent: AnalysisIntent) -> str:
        """Generate code for filtering."""
        if not intent.filter_conditions:
            return "# No filter conditions specified\nresult = df"
        
        filter_code_parts = []
        for condition in intent.filter_conditions:
            col = condition.get("column", "")
            op = condition.get("operator", "=")
            val = condition.get("value", "")
            
            if op == ">":
                filter_code_parts.append(f"df = df[df['{col}'] > {val}]")
            elif op == "<":
                filter_code_parts.append(f"df = df[df['{col}'] < {val}]")
            elif op == "=":
                filter_code_parts.append(f"df = df[df['{col}'] == '{val}']")
        
        code = f"""
# Filter data
{chr(10).join(filter_code_parts)}
result = df
"""
        return code
    
    def _generate_sort_code(self, intent: AnalysisIntent) -> str:
        """Generate code for sorting."""
        if not intent.sort_columns:
            return "# No sort columns specified\nresult = df"
        
        sort_cols = str(intent.sort_columns) if len(intent.sort_columns) > 1 else f"'{intent.sort_columns[0]}'"
        
        code = f"""
# Sort data
result = df.sort_values({sort_cols}, ascending={str(intent.sort_ascending)})
"""
        return code
    
    def _generate_group_by_code(self, intent: AnalysisIntent) -> str:
        """Generate code for group by analysis."""
        group_by = self._format_group_by(intent.group_by_columns)
        target_col = intent.target_columns[0] if intent.target_columns else "value"
        
        code = f"""
# Group by analysis
result = df.groupby({group_by})['{target_col}'].agg(['sum', 'mean', 'count']).reset_index()
result = result.sort_values('sum', ascending={str(intent.sort_ascending)})
"""
        return code
    
    def _generate_basic_analysis_code(self, intent: AnalysisIntent) -> str:
        """Generate basic analysis code."""
        code = """
# Basic analysis
result = df.describe()
"""
        return code
    
    def _generate_visualization_code(self, intent: AnalysisIntent) -> str:
        """Generate visualization code."""
        chart_type = intent.chart_type or "bar"
        
        if chart_type == "line":
            chart_params = "x=result.columns[0], y=result.columns[1]"
            title = f"{intent.analysis_type.value.title()} Analysis"
        elif chart_type == "bar":
            chart_params = "x=result.columns[0], y=result.columns[1]"
            title = f"{intent.analysis_type.value.title()} Analysis"
        elif chart_type == "pie":
            chart_params = "values=result.iloc[:, 1], names=result.iloc[:, 0]"
            title = f"{intent.analysis_type.value.title()} Distribution"
        else:
            chart_params = "x=result.columns[0], y=result.columns[1]"
            title = f"{intent.analysis_type.value.title()} Analysis"
        
        code = f"""
# Create visualization
fig = px.{chart_type}(result, {chart_params})
fig.update_layout(
    title='{title}',
    xaxis_title=result.columns[0],
    yaxis_title=result.columns[1] if len(result.columns) > 1 else 'Value'
)
fig.show()
"""
        return code
    
    def _generate_data_preview_code(self) -> str:
        """Generate code for data preview."""
        return """
# Data preview
print("Data shape:", df.shape)
print("\\nColumn names:", df.columns.tolist())
print("\\nData types:")
print(df.dtypes)
print("\\nFirst 5 rows:")
print(df.head())
print("\\nBasic statistics:")
print(df.describe())
"""
    
    def _generate_result_display_code(self, intent: AnalysisIntent) -> str:
        """Generate code for displaying results."""
        return """
# Display results
print("\\nAnalysis Results:")
print(result)
print("\\nResult shape:", result.shape)
"""
    
    def _format_group_by(self, group_by_columns: List[str]) -> str:
        """Format group by columns for code generation."""
        if not group_by_columns:
            return "[]"
        elif len(group_by_columns) == 1:
            return f"['{group_by_columns[0]}']"
        else:
            return str([f"'{col}'" for col in group_by_columns])
    
    def _generate_error_code(self, error_message: str) -> str:
        """Generate error handling code."""
        return f"""
# Error occurred during code generation
print("Error: {error_message}")
print("Please check your query and try again.")
"""
    
    def get_code_history(self) -> List[Dict[str, Any]]:
        """Get history of generated code."""
        return self.generated_code_history
    
    def validate_generated_code(self, code: str) -> bool:
        """Validate that generated code is syntactically correct."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False


