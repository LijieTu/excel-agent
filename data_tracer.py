"""
Data traceability system for tracking which columns are used in analysis.
"""
import pandas as pd
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from nlp_agent import AnalysisIntent

logger = logging.getLogger(__name__)

@dataclass
class ColumnUsage:
    """Represents usage of a column in analysis."""
    column_name: str
    usage_type: str  # 'target', 'group_by', 'filter', 'sort', 'time'
    operation: str
    value_range: Optional[tuple] = None
    null_count: int = 0
    unique_count: int = 0

@dataclass
class AnalysisTrace:
    """Represents a complete analysis trace."""
    analysis_id: str
    timestamp: datetime
    query: str
    file_path: str
    sheet_name: str
    intent: AnalysisIntent
    used_columns: List[ColumnUsage]
    generated_code: str
    execution_time: float
    result_shape: tuple
    success: bool
    error_message: Optional[str] = None

class DataTracer:
    """Tracks data usage and provides traceability information."""
    
    def __init__(self):
        self.analysis_history: List[AnalysisTrace] = []
        self.column_usage_stats: Dict[str, Dict[str, Any]] = {}
        self.file_metadata: Dict[str, Dict[str, Any]] = {}
    
    def trace_analysis(self, 
                      analysis_id: str,
                      query: str,
                      file_path: str,
                      sheet_name: str,
                      intent: AnalysisIntent,
                      df: pd.DataFrame,
                      generated_code: str,
                      execution_time: float,
                      result_df: pd.DataFrame,
                      success: bool = True,
                      error_message: Optional[str] = None) -> AnalysisTrace:
        """
        Trace an analysis and record column usage.
        
        Args:
            analysis_id: Unique identifier for the analysis
            query: Original user query
            file_path: Path to the Excel file
            sheet_name: Name of the sheet
            intent: Analysis intent
            df: Original DataFrame
            generated_code: Generated Python code
            execution_time: Time taken to execute
            result_df: Result DataFrame
            success: Whether analysis was successful
            error_message: Error message if failed
            
        Returns:
            AnalysisTrace object
        """
        # Extract column usage information
        used_columns = self._extract_column_usage(intent, df)
        
        # Create analysis trace
        trace = AnalysisTrace(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            query=query,
            file_path=file_path,
            sheet_name=sheet_name,
            intent=intent,
            used_columns=used_columns,
            generated_code=generated_code,
            execution_time=execution_time,
            result_shape=result_df.shape if success else (0, 0),
            success=success,
            error_message=error_message
        )
        
        # Store trace
        self.analysis_history.append(trace)
        
        # Update column usage statistics
        self._update_column_stats(used_columns, file_path, sheet_name)
        
        # Update file metadata
        self._update_file_metadata(file_path, sheet_name, df)
        
        logger.info(f"Analysis traced: {analysis_id} - {len(used_columns)} columns used")
        
        return trace
    
    def _extract_column_usage(self, intent: AnalysisIntent, df: pd.DataFrame) -> List[ColumnUsage]:
        """Extract column usage information from analysis intent and DataFrame."""
        used_columns = []
        
        # Track target columns
        for col in intent.target_columns:
            if col in df.columns:
                usage = ColumnUsage(
                    column_name=col,
                    usage_type='target',
                    operation=intent.analysis_type.value,
                    null_count=df[col].isnull().sum(),
                    unique_count=df[col].nunique()
                )
                used_columns.append(usage)
        
        # Track group by columns
        for col in intent.group_by_columns:
            if col in df.columns:
                usage = ColumnUsage(
                    column_name=col,
                    usage_type='group_by',
                    operation='group_by',
                    null_count=df[col].isnull().sum(),
                    unique_count=df[col].nunique()
                )
                used_columns.append(usage)
        
        # Track filter columns
        for condition in intent.filter_conditions:
            col = condition.get('column', '')
            if col in df.columns:
                usage = ColumnUsage(
                    column_name=col,
                    usage_type='filter',
                    operation=f"filter_{condition.get('operator', '=')}",
                    null_count=df[col].isnull().sum(),
                    unique_count=df[col].nunique()
                )
                used_columns.append(usage)
        
        # Track sort columns
        for col in intent.sort_columns:
            if col in df.columns:
                usage = ColumnUsage(
                    column_name=col,
                    usage_type='sort',
                    operation=f"sort_{'asc' if intent.sort_ascending else 'desc'}",
                    null_count=df[col].isnull().sum(),
                    unique_count=df[col].nunique()
                )
                used_columns.append(usage)
        
        # Track time column
        if intent.time_column and intent.time_column in df.columns:
            usage = ColumnUsage(
                column_name=intent.time_column,
                usage_type='time',
                operation='time_series',
                null_count=df[intent.time_column].isnull().sum(),
                unique_count=df[intent.time_column].nunique()
            )
            used_columns.append(usage)
        
        return used_columns
    
    def _update_column_stats(self, used_columns: List[ColumnUsage], file_path: str, sheet_name: str) -> None:
        """Update column usage statistics."""
        key = f"{file_path}_{sheet_name}"
        
        if key not in self.column_usage_stats:
            self.column_usage_stats[key] = {}
        
        for usage in used_columns:
            col_key = usage.column_name
            if col_key not in self.column_usage_stats[key]:
                self.column_usage_stats[key][col_key] = {
                    'usage_count': 0,
                    'usage_types': set(),
                    'operations': set(),
                    'last_used': None
                }
            
            stats = self.column_usage_stats[key][col_key]
            stats['usage_count'] += 1
            stats['usage_types'].add(usage.usage_type)
            stats['operations'].add(usage.operation)
            stats['last_used'] = datetime.now()
    
    def _update_file_metadata(self, file_path: str, sheet_name: str, df: pd.DataFrame) -> None:
        """Update file metadata."""
        key = f"{file_path}_{sheet_name}"
        
        self.file_metadata[key] = {
            'file_path': file_path,
            'sheet_name': sheet_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'last_accessed': datetime.now(),
            'analysis_count': len([t for t in self.analysis_history 
                                 if t.file_path == file_path and t.sheet_name == sheet_name])
        }
    
    def get_column_usage_report(self, file_path: str, sheet_name: str = None) -> Dict[str, Any]:
        """
        Get column usage report for a specific file/sheet.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet (if None, returns for all sheets)
            
        Returns:
            Dictionary containing column usage information
        """
        if sheet_name:
            key = f"{file_path}_{sheet_name}"
            return self.column_usage_stats.get(key, {})
        else:
            # Return for all sheets in the file
            file_stats = {}
            for key, stats in self.column_usage_stats.items():
                if key.startswith(file_path):
                    sheet_name = key.split('_', 1)[1] if '_' in key else 'Sheet1'
                    file_stats[sheet_name] = stats
            return file_stats
    
    def get_analysis_summary(self, analysis_id: str = None) -> Dict[str, Any]:
        """
        Get analysis summary.
        
        Args:
            analysis_id: Specific analysis ID (if None, returns overall summary)
            
        Returns:
            Dictionary containing analysis summary
        """
        if analysis_id:
            # Return specific analysis trace
            for trace in self.analysis_history:
                if trace.analysis_id == analysis_id:
                    return {
                        'analysis_id': trace.analysis_id,
                        'timestamp': trace.timestamp.isoformat(),
                        'query': trace.query,
                        'file_path': trace.file_path,
                        'sheet_name': trace.sheet_name,
                        'analysis_type': trace.intent.analysis_type.value,
                        'used_columns': [
                            {
                                'name': col.column_name,
                                'type': col.usage_type,
                                'operation': col.operation,
                                'null_count': col.null_count,
                                'unique_count': col.unique_count
                            }
                            for col in trace.used_columns
                        ],
                        'execution_time': trace.execution_time,
                        'result_shape': trace.result_shape,
                        'success': trace.success,
                        'error_message': trace.error_message
                    }
            return {}
        else:
            # Return overall summary
            total_analyses = len(self.analysis_history)
            successful_analyses = len([t for t in self.analysis_history if t.success])
            failed_analyses = total_analyses - successful_analyses
            
            # Get unique files and columns used
            unique_files = set(t.file_path for t in self.analysis_history)
            all_columns = set()
            for trace in self.analysis_history:
                for col in trace.used_columns:
                    all_columns.add(col.column_name)
            
            return {
                'total_analyses': total_analyses,
                'successful_analyses': successful_analyses,
                'failed_analyses': failed_analyses,
                'success_rate': successful_analyses / total_analyses if total_analyses > 0 else 0,
                'unique_files': len(unique_files),
                'unique_columns_used': len(all_columns),
                'average_execution_time': sum(t.execution_time for t in self.analysis_history) / total_analyses if total_analyses > 0 else 0
            }
    
    def get_data_lineage(self, column_name: str, file_path: str = None) -> List[Dict[str, Any]]:
        """
        Get data lineage for a specific column.
        
        Args:
            column_name: Name of the column
            file_path: Optional file path to filter by
            
        Returns:
            List of analysis traces that used this column
        """
        lineage = []
        
        for trace in self.analysis_history:
            if file_path and trace.file_path != file_path:
                continue
            
            for col in trace.used_columns:
                if col.column_name == column_name:
                    lineage.append({
                        'analysis_id': trace.analysis_id,
                        'timestamp': trace.timestamp.isoformat(),
                        'query': trace.query,
                        'usage_type': col.usage_type,
                        'operation': col.operation,
                        'file_path': trace.file_path,
                        'sheet_name': trace.sheet_name
                    })
                    break
        
        return lineage
    
    def export_trace_data(self, file_path: str) -> None:
        """
        Export trace data to JSON file.
        
        Args:
            file_path: Path to export file
        """
        export_data = {
            'analysis_history': [
                {
                    'analysis_id': trace.analysis_id,
                    'timestamp': trace.timestamp.isoformat(),
                    'query': trace.query,
                    'file_path': trace.file_path,
                    'sheet_name': trace.sheet_name,
                    'analysis_type': trace.intent.analysis_type.value,
                    'used_columns': [
                        {
                            'name': col.column_name,
                            'type': col.usage_type,
                            'operation': col.operation,
                            'null_count': col.null_count,
                            'unique_count': col.unique_count
                        }
                        for col in trace.used_columns
                    ],
                    'execution_time': trace.execution_time,
                    'result_shape': trace.result_shape,
                    'success': trace.success,
                    'error_message': trace.error_message
                }
                for trace in self.analysis_history
            ],
            'column_usage_stats': self.column_usage_stats,
            'file_metadata': self.file_metadata
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Trace data exported to {file_path}")
    
    def clear_history(self) -> None:
        """Clear all trace history."""
        self.analysis_history.clear()
        self.column_usage_stats.clear()
        self.file_metadata.clear()
        logger.info("Trace history cleared")
