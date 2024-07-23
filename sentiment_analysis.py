import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)



#conveting audio to text
import speech_recognition as sr
from pydub import AudioSegment

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    
    # Split audio into chunks and apply speech recognition
    chunk_length_ms = 30000  # 30 seconds
    chunks = make_chunks(audio, chunk_length_ms)
    
    transcript = []
    for i, chunk in enumerate(chunks):
        chunk_filename = f'chunk{i}.wav'
        chunk.export(chunk_filename, format='wav')
        
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_listened)
                timestamp = (i * chunk_length_ms) / 1000
                transcript.append({'timestamp': timestamp, 'text': text})
            except sr.UnknownValueError:
                print(f'Chunk {i} could not be transcribed')
    
    return transcript




import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load the tokenizer and model

def make_transcript(audio_path):
    load_directory = '/Users/shivamgoswami/Documents/video_extraction/SJ-Ra-Re-Punctuate'

    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(load_directory)
    model = TFT5ForConditionalGeneration.from_pretrained('SJ-Ray/Re-Punctuate')

    transcript = transcribe_audio(audio_path)
    punctuated_txt=[]
    for i in transcript:
        inputs = tokenizer.encode("punctuate: " + i["text"], return_tensors="tf") 
        result = model.generate(inputs)

        decoded_output = tokenizer.decode(result[0], skip_special_tokens=True)

        punctuated_txt.append(decoded_output)

    transcript="".join(punctuated_txt)
    transcript=transcript.split(".")
    transcript = [sentence for sentence in transcript if len(sentence.split()) >= 7]
    return transcript



from transformers import pipeline

# Specify the model explicitly



def sentiment_(video_path):
    audio_path="".join(video_path.split("/"))[:-1]+".wav"
    load_directory="/Users/shivamgoswami/Documents/video_extraction/bert"
    tokenizer = AutoTokenizer.from_pretrained(load_directory)
    model = AutoModelForSequenceClassification.from_pretrained(load_directory)
    # Load the sentiment-analysis pipeline with the specified model
    sentiment_analyzer = pipeline('sentiment-analysis',  model=model, tokenizer=tokenizer)
    extract_audio_from_video(video_path, audio_path)
    transcript=make_transcript(audio_path)
    sentiments = sentiment_analyzer(transcript)

    result=[]
    for i, sentiment in enumerate(sentiments):
        #print(f"{transcript[i]} Segment {i+1}: {sentiment['label']} (score: {sentiment['score']:.2f})")
        result.append([transcript[i],"----",sentiment["label"],"----",sentiment["score"]])
    return result


    