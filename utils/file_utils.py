"""
Utility functions for file management and visualization.
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging
from config import Config

logger = logging.getLogger(__name__)

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    try:
        stat = os.stat(file_path)
        return {
            'name': os.path.basename(file_path),
            'size': stat.st_size,
            'modified': pd.Timestamp.fromtimestamp(stat.st_mtime),
            'extension': os.path.splitext(file_path)[1].lower()
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return {}

def validate_excel_file(file_path: str) -> bool:
    """
    Validate that a file is a valid Excel file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid Excel file, False otherwise
    """
    try:
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.xlsx', '.xls', '.csv']:
            return False
        
        # Try to read the file
        if ext == '.csv':
            pd.read_csv(file_path, nrows=1)
        else:
            pd.read_excel(file_path, nrows=1, engine='openpyxl')
        
        return True
    except Exception as e:
        logger.error(f"Error validating Excel file {file_path}: {str(e)}")
        return False

def create_sample_excel_files() -> None:
    """Create sample Excel files for demonstration."""
    sample_dir = Config.DATA_DIR
    
    # Sample sales data
    sales_data = {
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Region': ['North', 'South', 'East', 'West'] * 25,
        'Product': ['A', 'B', 'C', 'D', 'E'] * 20,
        'Sales': [1000 + i * 10 + (i % 4) * 100 for i in range(100)],
        'Quantity': [10 + i % 20 for i in range(100)],
        'Customer_ID': [f'CUST_{i:03d}' for i in range(100)]
    }
    
    sales_df = pd.DataFrame(sales_data)
    sales_file = os.path.join(sample_dir, 'sales_data.xlsx')
    sales_df.to_excel(sales_file, index=False)
    
    # Sample employee data
    employee_data = {
        'Employee_ID': [f'EMP_{i:03d}' for i in range(50)],
        'Name': [f'Employee_{i}' for i in range(50)],
        'Department': ['HR', 'Finance', 'IT', 'Marketing', 'Sales'] * 10,
        'Salary': [50000 + i * 1000 + (i % 5) * 5000 for i in range(50)],
        'Experience_Years': [1 + i % 10 for i in range(50)],
        'Manager': [f'Manager_{i % 5}' for i in range(50)]
    }
    
    employee_df = pd.DataFrame(employee_data)
    employee_file = os.path.join(sample_dir, 'employee_data.xlsx')
    employee_df.to_excel(employee_file, index=False)
    
    logger.info(f"Sample Excel files created in {sample_dir}")

def get_available_files() -> List[Dict[str, Any]]:
    """
    Get list of available Excel files in the data directory.
    
    Returns:
        List of file information dictionaries
    """
    files = []
    data_dir = Config.DATA_DIR
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        create_sample_excel_files()
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path) and validate_excel_file(file_path):
            file_info = get_file_info(file_path)
            file_info['path'] = file_path
            files.append(file_info)
    
    return files


