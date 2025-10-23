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
        # Simple text area for transcript input (no complex HTML component)
        st.write("**📝 Voice Transcript:**")
        transcript_input = st.text_area(
            "Paste your voice transcript here:",
            height=100,
            placeholder="After recording, paste your transcript here to analyze it...",
            key="voice_transcript_simple"
        )

        if transcript_input and transcript_input.strip():
            return transcript_input.strip()

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


