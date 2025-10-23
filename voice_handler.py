"""
Voice input processing functionality for real-time speech-to-text conversion.
"""
import openai
import asyncio
import websockets
import json
import base64
import io
import logging
from typing import Optional, Callable, Dict, Any
import streamlit as st
from config import Config

logger = logging.getLogger(__name__)

class VoiceHandler:
    """Handles voice input processing using multiple methods."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.audio_buffer = []
        self.is_recording = False
    
    def process_audio_file(self, audio_file) -> str:
        """
        Process uploaded audio file using OpenAI Whisper API.
        
        Args:
            audio_file: Uploaded audio file from Streamlit
            
        Returns:
            Transcribed text
        """
        try:
            # Read audio file
            audio_bytes = audio_file.read()
            
            # Create a file-like object
            audio_file_obj = io.BytesIO(audio_bytes)
            audio_file_obj.name = "audio.wav"
            
            # Transcribe using Whisper
            transcript = self.client.audio.transcriptions.create(
                model=Config.WHISPER_MODEL,
                file=audio_file_obj,
                response_format="text"
            )
            
            logger.info(f"Audio transcription successful: {len(transcript)} characters")
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            return f"Error transcribing audio: {str(e)}"
    
    def process_audio_bytes(self, audio_bytes: bytes) -> str:
        """
        Process audio bytes using OpenAI Whisper API.
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Create a file-like object
            audio_file_obj = io.BytesIO(audio_bytes)
            audio_file_obj.name = "audio.wav"
            
            # Transcribe using Whisper
            transcript = self.client.audio.transcriptions.create(
                model=Config.WHISPER_MODEL,
                file=audio_file_obj,
                response_format="text"
            )
            
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"Error processing audio bytes: {str(e)}")
            return f"Error transcribing audio: {str(e)}"
    
    def validate_audio_file(self, audio_file) -> bool:
        """
        Validate uploaded audio file.
        
        Args:
            audio_file: Uploaded audio file
            
        Returns:
            True if valid, False otherwise
        """
        if audio_file is None:
            return False
        
        # Check file size (max 25MB for Whisper API)
        max_size = 25 * 1024 * 1024  # 25MB
        if audio_file.size > max_size:
            return False
        
        # Check file type
        allowed_types = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/webm']
        if audio_file.type not in allowed_types:
            return False
        
        return True
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats."""
        return ['WAV', 'MP3', 'MPEG', 'M4A', 'WEBM']
    
    def create_voice_input_ui(self) -> Optional[str]:
        """
        Create voice input UI components for Streamlit.
        
        Returns:
            Transcribed text if successful, None otherwise
        """
        st.subheader("🎤 Voice Input")
        
        # Method 1: File Upload
        st.write("**Method 1: Upload Audio File**")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'webm'],
            help=f"Supported formats: {', '.join(self.get_supported_formats())}. Max size: 25MB"
        )
        
        if audio_file is not None:
            if self.validate_audio_file(audio_file):
                if st.button("🎵 Transcribe Audio"):
                    with st.spinner("Transcribing audio..."):
                        transcript = self.process_audio_file(audio_file)
                        if transcript and not transcript.startswith("Error"):
                            st.success("✅ Transcription successful!")
                            return transcript
                        else:
                            st.error(f"❌ {transcript}")
            else:
                st.error("❌ Invalid audio file. Please check format and size.")
        
        # Method 2: Browser Speech Recognition (JavaScript)
        st.write("**Method 2: Real-time Voice Input**")
        st.write("Click the microphone button below to start recording:")
        
        # Create HTML/JavaScript for Web Speech API
        voice_input_html = """
        <div id="voice-input-container">
            <button id="start-recording" onclick="startRecording()" style="
                background-color: #ff4b4b;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            ">🎤 Start Recording</button>
            
            <button id="stop-recording" onclick="stopRecording()" style="
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
                display: none;
            ">⏹️ Stop Recording</button>
            
            <div id="status" style="margin: 10px; font-weight: bold;"></div>
            <div id="transcript" style="margin: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;"></div>
        </div>

        <script>
        let recognition;
        let isRecording = false;

        function startRecording() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'en-US';
                
                recognition.onstart = function() {
                    isRecording = true;
                    document.getElementById('start-recording').style.display = 'none';
                    document.getElementById('stop-recording').style.display = 'inline-block';
                    document.getElementById('status').textContent = '🎤 Listening...';
                };
                
                recognition.onresult = function(event) {
                    let transcript = '';
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        transcript += event.results[i][0].transcript;
                    }
                    document.getElementById('transcript').textContent = transcript;
                };
                
                recognition.onerror = function(event) {
                    document.getElementById('status').textContent = '❌ Error: ' + event.error;
                    stopRecording();
                };
                
                recognition.onend = function() {
                    stopRecording();
                };
                
                recognition.start();
            } else {
                document.getElementById('status').textContent = '❌ Speech recognition not supported in this browser';
            }
        }

        function stopRecording() {
            if (recognition && isRecording) {
                recognition.stop();
                isRecording = false;
                document.getElementById('start-recording').style.display = 'inline-block';
                document.getElementById('stop-recording').style.display = 'none';
                document.getElementById('status').textContent = '✅ Recording stopped';
                
                // Send transcript to Streamlit
                const transcript = document.getElementById('transcript').textContent;
                if (transcript.trim()) {
                    // Use Streamlit's session state to pass data
                    parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: transcript
                    }, '*');
                }
            }
        }
        </script>
        """
        
        st.components.v1.html(voice_input_html, height=200)
        
        # Check for transcript in session state
        if 'voice_transcript' in st.session_state:
            transcript = st.session_state.voice_transcript
            if transcript:
                st.success(f"🎤 Voice input received: {transcript}")
                return transcript
        
        return None
    
    def setup_websocket_server(self, port: int = 8765) -> None:
        """
        Setup WebSocket server for real-time voice input.
        
        Args:
            port: Port number for WebSocket server
        """
        async def handle_voice_data(websocket, path):
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data['type'] == 'audio':
                        # Process audio data
                        audio_bytes = base64.b64decode(data['audio'])
                        transcript = self.process_audio_bytes(audio_bytes)
                        
                        # Send back transcript
                        response = {
                            'type': 'transcript',
                            'text': transcript,
                            'success': True
                        }
                        await websocket.send(json.dumps(response))
                    
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                error_response = {
                    'type': 'error',
                    'message': str(e),
                    'success': False
                }
                await websocket.send(json.dumps(error_response))
        
        # Start WebSocket server
        start_server = websockets.serve(handle_voice_data, "localhost", port)
        asyncio.get_event_loop().run_until_complete(start_server)
        logger.info(f"WebSocket server started on port {port}")
    
    def cleanup_audio_buffer(self) -> None:
        """Clean up audio buffer."""
        self.audio_buffer.clear()
    
    def get_recording_status(self) -> bool:
        """Get current recording status."""
        return self.is_recording
    
    def set_recording_status(self, status: bool) -> None:
        """Set recording status."""
        self.is_recording = status


