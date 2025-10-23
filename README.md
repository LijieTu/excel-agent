# Intelligent Excel Agent

An intelligent Excel analysis system that understands natural language queries and automatically performs data analysis tasks. The system can parse Excel files, generate Python analysis code, and provide data traceability.

## Features

- **Natural Language Processing**: Understand queries in English and Chinese
- **Voice Input**: Real-time voice input via WebSocket and file upload
- **Automatic Code Generation**: Generates Python pandas/numpy code based on analysis intent
- **Data Traceability**: Tracks which columns are used in each analysis
- **Interactive Visualizations**: Creates charts using Plotly
- **Excel File Processing**: Handles complex spreadsheets and reshapes them into 2D tables

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd excel-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env_example.txt .env
```

4. Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Select an Excel file from the sidebar

4. Use one of the input methods:
   - **Voice Input**: Click the microphone button to record audio
   - **Text Query**: Type your analysis request in natural language

5. View the generated code, results, and visualizations

## Example Queries

- "Show me the total sales by region"
- "What are the average salaries by department?"
- "Create a trend chart of sales over time"
- "Which products have the highest sales?"
- "Show me the correlation between salary and experience"

## Project Structure

```
excel-agent/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── config.py                       # Configuration settings
├── excel_processor.py             # Excel file processing utilities
├── nlp_agent.py                   # Natural language processing
├── code_generator.py              # Python code generation
├── data_tracer.py                 # Data column tracking
├── voice_handler.py              # Voice input processing
├── utils/
│   ├── __init__.py
│   ├── file_utils.py              # File management utilities
│   └── visualization.py           # Chart generation helpers
├── static/
│   ├── css/
│   │   └── custom.css             # Custom styling
│   └── js/
│       └── voice_recorder.js      # JavaScript for voice input
├── data/
│   └── sample_excel/              # Predefined Excel files directory
└── README.md                      # This file
```

## Configuration

The application can be configured through the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4)
- `DEBUG`: Enable debug mode (default: True)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 50)
- `ENABLE_VOICE_INPUT`: Enable voice input features (default: True)

## Voice Input

The system supports multiple voice input methods:

1. **File Upload**: Upload audio files (WAV, MP3, M4A, WEBM)
2. **Real-time Recording**: Use browser's Web Speech API
3. **WebSocket**: Real-time audio streaming (advanced)

## Data Traceability

The system tracks:
- Which columns are used in each analysis
- Analysis execution time and success rate
- Data lineage and usage statistics
- Export capabilities for trace data

## Supported File Formats

- Excel files (.xlsx, .xls)
- CSV files (.csv)
- Audio files for voice input (.wav, .mp3, .m4a, .webm)

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Make sure your API key is correctly set in the `.env` file
2. **File Loading Error**: Check that the Excel file is not corrupted and is in a supported format
3. **Voice Input Not Working**: Ensure your browser supports Web Speech API (Chrome/Edge recommended)

### Logs

Check the console output for detailed error messages and logging information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 and Whisper API
- Streamlit for the web framework
- Plotly for visualizations
- Pandas for data processing


