"""
Configuration settings for the Excel Agent application.
"""
import os
from dotenv import load_dotenv

# Load environment variables with override
load_dotenv(override=True)

class Config:
    """Application configuration class."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Application Settings
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    SUPPORTED_FILE_TYPES = os.getenv('SUPPORTED_FILE_TYPES', 'xlsx,xls,csv').split(',')
    
    # Voice Input Settings
    ENABLE_VOICE_INPUT = os.getenv('ENABLE_VOICE_INPUT', 'True').lower() == 'true'
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'whisper-1')
    
    # Data Processing Settings
    MAX_ROWS_PREVIEW = int(os.getenv('MAX_ROWS_PREVIEW', '1000'))
    DEFAULT_CHART_TYPE = os.getenv('DEFAULT_CHART_TYPE', 'line')
    
    # File Paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'sample_excel')
    STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            # Try to read directly from .env file as fallback
            env_file = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            cls.OPENAI_API_KEY = line.split('=', 1)[1].strip()
                            break
        
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        
        # Create directories if they don't exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.STATIC_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.STATIC_DIR, 'css'), exist_ok=True)
        os.makedirs(os.path.join(cls.STATIC_DIR, 'js'), exist_ok=True)


