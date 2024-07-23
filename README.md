# Video Sentiment Analysis

This project performs sentiment analysis on audio extracted from videos. It involves extracting audio from video files, transcribing the audio to text, punctuating the transcript, and then analyzing the sentiment of the transcribed text.

## Files

### `app.py`
This script serves as the entry point for the application. It sets up and executes the main functionality of the project, including processing video files and analyzing sentiment.

### `sentiment_analysis.py`
This script contains the core functionality of the project:
- **`extract_audio_from_video(video_path, audio_path)`**: Extracts audio from a video file and saves it as a `.wav` file.
- **`transcribe_audio(audio_path)`**: Converts the extracted audio into text by splitting it into chunks and using Google’s speech recognition service.
- **`make_transcript(audio_path)`**: Transcribes the audio and punctuates the text using a T5 model for punctuation restoration.
- **`sentiment_(video_path)`**: Analyzes the sentiment of the transcribed and punctuated text using a sentiment analysis model.

### `test.ipynb`
A Jupyter Notebook file used for testing the functionalities of the scripts and experimenting with different aspects of the project.

### `test2.ipynb`
Another Jupyter Notebook file,  for further testing and validation of different features or models used in the project.

### `videoplayback.mp4`
A sample video file used for testing the extraction of audio and subsequent sentiment analysis.

## Dependencies

- **Transformers**: For sentiment analysis and punctuation restoration.
- **Pydub**: For audio processing and chunking.
- **SpeechRecognition**: For converting speech to text.
- **MoviePy**: For handling video files.
- **Torch**: For running PyTorch models.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/iamshivamgoswami/video_sentiment.git
   cd video_sentiment
