"""
Excel file processing utilities for reshaping complex spreadsheets into 2D tables.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import openpyxl
from openpyxl.utils import get_column_letter
import logging

logger = logging.getLogger(__name__)

class ExcelProcessor:
    """Handles Excel file processing and data reshaping."""
    
    def __init__(self):
        self.processed_files = {}
        self.column_metadata = {}
    
    def load_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load Excel file and return all sheets as DataFrames.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary mapping sheet names to DataFrames
        """
        try:
            # Read all sheets
            excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            
            processed_sheets = {}
            for sheet_name, df in excel_data.items():
                # Clean and reshape the data
                cleaned_df = self._clean_dataframe(df)
                processed_sheets[sheet_name] = cleaned_df
                
                # Store metadata
                self.column_metadata[f"{file_path}_{sheet_name}"] = self._extract_column_metadata(cleaned_df)
            
            self.processed_files[file_path] = processed_sheets
            return processed_sheets
            
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {str(e)}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and reshape DataFrame to ensure it's a proper 2D table.
        
        Args:
            df: Raw DataFrame from Excel
            
        Returns:
            Cleaned DataFrame
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Handle multi-level headers
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns
            df.columns = ['_'.join(str(col).strip() for col in df.columns if str(col) != 'Unnamed: level_0')
                         for df.columns in df.columns]
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Remove unnamed columns
        df.columns = [col if not col.startswith('Unnamed:') else f'Column_{i}' 
                     for i, col in enumerate(df.columns)]
        
        # Convert data types intelligently
        df = self._convert_data_types(df)
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently convert data types in the DataFrame.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            DataFrame with converted data types
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
                
                # Try to convert to datetime
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        pass
        
        return df
    
    def _extract_column_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract metadata about columns in the DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing column metadata
        """
        metadata = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': df.nunique().to_dict(),
            'shape': df.shape,
            'sample_data': df.head(3).to_dict('records')
        }
        
        # Add statistical summaries for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metadata['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        return metadata
    
    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """
        Get a summary of the Excel file structure.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing file summary
        """
        if file_path not in self.processed_files:
            self.load_excel_file(file_path)
        
        summary = {
            'file_path': file_path,
            'sheets': list(self.processed_files[file_path].keys()),
            'total_sheets': len(self.processed_files[file_path])
        }
        
        # Add sheet details
        sheet_details = {}
        for sheet_name, df in self.processed_files[file_path].items():
            sheet_details[sheet_name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'has_nulls': df.isnull().any().any()
            }
        
        summary['sheet_details'] = sheet_details
        return summary
    
    def get_column_info(self, file_path: str, sheet_name: str = None) -> Dict[str, Any]:
        """
        Get detailed information about columns in a specific sheet.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet (if None, uses first sheet)
            
        Returns:
            Dictionary containing column information
        """
        if file_path not in self.processed_files:
            self.load_excel_file(file_path)
        
        if sheet_name is None:
            sheet_name = list(self.processed_files[file_path].keys())[0]
        
        key = f"{file_path}_{sheet_name}"
        return self.column_metadata.get(key, {})
    
    def reshape_for_analysis(self, df: pd.DataFrame, target_format: str = 'long') -> pd.DataFrame:
        """
        Reshape DataFrame for specific analysis needs.
        
        Args:
            df: DataFrame to reshape
            target_format: Target format ('long', 'wide', 'pivot')
            
        Returns:
            Reshaped DataFrame
        """
        if target_format == 'long':
            # Convert wide format to long format
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                return df.melt(id_vars=[col for col in df.columns if col not in numeric_cols],
                              value_vars=numeric_cols,
                              var_name='metric', value_name='value')
        
        return df


