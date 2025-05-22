import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import subprocess
import os
import ffmpeg
import urllib.request
import glob
import requests
import logging
import sys
import time
from transformers import pipeline, logging as hf_logging
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL
import threading

hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

API_KEY = st.secrets["API_KEY"]

@st.cache_resource
def get_whisper_model():
    return WhisperModel("tiny", compute_type="int8", device="cpu")

@st.cache_resource
def get_classifier():
    return pipeline("audio-classification", model="dima806/english_accents_classification")

def transcribe_with_timeout(audio_path, timeout=90):
    result = {}

    def worker():
        try:
            model = get_whisper_model()
            segments_gen, info = model.transcribe(audio_path)
            result["segments"] = list(segments_gen)
            result["info"] = info
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError("Transcription timed out.")

    if "error" in result:
        raise RuntimeError(result["error"])

    return result["segments"], result["info"]

# Download and convert
def download_audio(input_path_or_url, output_path="audio.wav"):
    temp_pattern = "temp_audio.%(ext)s"
    if "youtube.com" in input_path_or_url or "youtu.be" in input_path_or_url:
        st.info("📥 Downloading YouTube audio...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_pattern,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([input_path_or_url])
        temp_files = glob.glob("temp_audio.*")
        if not temp_files:
            raise FileNotFoundError("Download failed.")
        temp_video = temp_files[0]
    elif input_path_or_url.startswith("http"):
        st.info("🌐 Downloading direct video...")
        urllib.request.urlretrieve(input_path_or_url, "temp_video.mp4")
        temp_video = "temp_video.mp4"
    elif os.path.exists(input_path_or_url):
        st.info("📁 Using local file...")
        temp_video = input_path_or_url
    else:
        raise ValueError("Unsupported input type")

    st.info("🔄 Converting to WAV...")
    try:
        ffmpeg.input(temp_video).output(output_path, ac=1, ar=16000).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        st.error(f"❌ ffmpeg error:\n{e.stderr.decode()}")
        raise

    if temp_video != input_path_or_url and os.path.exists(temp_video):
        os.remove(temp_video)

    return output_path

def trim_audio(input_path, output_path="trimmed_audio.wav", duration=60):
    try:
        ffmpeg.input(input_path).output(output_path, t=duration).run(
            overwrite_output=True, capture_stdout=True, capture_stderr=True
        )
        return output_path
    except ffmpeg.Error as e:
        st.error(f"❌ Failed to trim audio:\n{e.stderr.decode()}")
        raise

# Transcribe
def transcribe_audio(audio_path):
    st.info("📝 Transcribing...")
    try:
        #model = get_whisper_model()
        #segments_gen, info = model.transcribe(audio_path)
        segments, info = transcribe_with_timeout(audio_path)
        #segments = list(segments_gen)  # materialize fully

        if info.duration > 300:
            st.error("❌ This audio is too long. Please use a clip under 5 minutes.")
            return ""

    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return ""

    full_text = ""
    progress = st.progress(0, text="Processing transcription...")
    #segments = list(segments)
    total = len(segments)
    for i, segment in enumerate(segments):
        full_text += segment.text.strip() + "\n"
        progress.progress((i + 1) / total)
    progress.empty()
    return full_text

# Accent classifier
def classify_accent(audio_path):
    st.info("🔍 Classifying accent...")

    progress = st.progress(0, text="Trimming audio to 60 seconds...")
    try:
        # Trim to first 60 seconds
        trimmed_audio = trim_audio(audio_path)

        progress.progress(30, text="Loading model...")
        classifier = get_classifier()

        # Simulate progress visually
        for i in range(30, 90, 10):
            time.sleep(0.2)
            progress.progress(i / 100.0, text="Classifying...")

        result = classifier(trimmed_audio)
        progress.progress(100, text="Done!")
        time.sleep(0.2)
        progress.empty()

        # Delete the temporary trimmed audio file
        if os.path.exists(trimmed_audio):
            os.remove(trimmed_audio)

        return result[0]["label"], result[0]["score"]

    except Exception as e:
        progress.empty()
        st.error(f"❌ Accent classification failed: {e}")
        return "Unknown", 0.0

# Gemini
def analyze_transcript_with_gemini(transcript_text, api_key):
    st.info("🤖 Evaluating with Gemini...")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt_text = f"""
This is a transcript of a spoken English video. Please do the following:
1. Rate the speaker’s English fluency on a scale from 0 to 100%.
2. Provide a short 2–3 sentence evaluation of their speaking quality.

Transcript:
{transcript_text[:1500]}
"""
    data = {
        "contents": [{"parts": [{"text": prompt_text}]}]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"❌ Gemini API error: {e}"

# Streamlit UI
st.title("🗣️ English Accent & Fluency Evaluator")
video_input = st.text_input("Paste a YouTube URL or video link:")

if st.button("Analyze"):
    if not video_input:
        st.warning("Please enter a video URL.")
    else:
        try:
            with st.status("📥 Downloading and converting audio...", expanded=True):
                audio_path = download_audio(video_input)
                st.success("✅ Audio downloaded and converted.")

            with st.status("📝 Transcribing audio...", expanded=True):
                transcript = transcribe_audio(audio_path)
                if not transcript:
                    raise RuntimeError("Transcription returned empty text.")
                st.success("✅ Transcription completed.")

            with st.status("🔍 Classifying accent...", expanded=True):
                accent, confidence = classify_accent(audio_path)
                st.success(f"✅ Accent: {accent.capitalize()} ({confidence:.2%})")

            with st.status("🤖 Asking Gemini to evaluate...", expanded=True):
                summary = analyze_transcript_with_gemini(transcript, api_key=API_KEY)
                st.success("✅ Gemini response received.")

            st.markdown("## 📄 Transcript")
            st.text_area("Transcript", transcript, height=200)

            st.markdown("## 🧠 Gemini Evaluation")
            st.write(summary)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
