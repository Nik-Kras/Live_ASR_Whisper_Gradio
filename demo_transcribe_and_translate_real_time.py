import gradio as gr
from faster_whisper import WhisperModel
import numpy as np
from scipy import signal
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import nltk
from transformers import pipeline
import time

# Download the necessary NLTK data
nltk.download('punkt')

class FasterWhisperStreamingWithTranslation:
    BEAM_SIZE = 5
    NO_SPEECH_PROBABILITY = 0.9
    MAX_SENTENCES = 2

    def __init__(self, model_size_or_path: str ="large-v2", sample_rate: int = 16_000, device: str ="auto",
                 update_period: float = 0.7, auto_update_period: bool = False):
        self.model = WhisperModel(model_size_or_path=model_size_or_path, device=device, compute_type="float16")
        # Initialize translation models
        self.es_to_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        self.en_to_es_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

        self.update_event_buffer = np.array([])
        self.input_audio_buffer = np.array([])
        self.processed_audio_length = 0
        self.original_update_period = update_period
        self.update_period = update_period
        self.auto_update_period = auto_update_period
        self.sampling_rate = sample_rate
        self.prompt = ""
        self.predicted_text = ""
        self.translated_text = ""
        self.log_messages = []

    def add_log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        if len(self.log_messages) > 10:
            self.log_messages = self.log_messages[-10:]
        return "\n".join(self.log_messages)

    def translate_text(self, text: str, source_lang: str) -> str:
        if not text.strip():
            return "", ""

        start_time = time.time()
        sentences = sent_tokenize(text)
        if source_lang == "es":
            translations = self.es_to_en_model(sentences)
        else:  # source_lang == "en"
            translations = self.en_to_es_model(sentences)

        translation_time = time.time() - start_time
        log_message = f"Translation ({source_lang} → {'en' if source_lang == 'es' else 'es'}) took {translation_time:.2f} seconds"
        logs = self.add_log(log_message)

        return " ".join(t['translation_text'] for t in translations), logs

    def listen(self, audio_chunk, source_lang: str, state: Dict):
        self.update_event_buffer = np.concatenate((self.update_event_buffer, audio_chunk))

        if len(self.update_event_buffer) > int(self.update_period * self.sampling_rate):
            new_text, transcription_logs = self.update(source_lang)

            current_transcription = state.get('transcription', '')
            if new_text.strip():
                if not current_transcription:
                    current_transcription = new_text
                else:
                    current_transcription = f"{current_transcription} {new_text}"

                # Translate the new text
                new_translation, translation_logs = self.translate_text(current_transcription, source_lang)

                # Update state
                state = {
                    'transcription': current_transcription,
                    'translation': new_translation,
                    'logs': self.log_messages
                }

        return state, state.get('transcription', ''), state.get('translation', ''), "\n".join(self.log_messages)

    def update(self, source_lang: str):
        # Add new audio to the buffer
        self.input_audio_buffer = np.concatenate((self.input_audio_buffer, self.update_event_buffer))
        self.update_event_buffer = np.array([])

        # Get new transcription
        time_stamped_transcription, logs = self.transcribe(source_lang)

        # Get only the new text from unprocessed audio
        new_text = self.extract_new_text(time_stamped_transcription)

        # Update processed audio length
        if time_stamped_transcription:
            last_word = time_stamped_transcription[-1]
            self.processed_audio_length = max(self.processed_audio_length,
                                           int(last_word['time_end'] * self.sampling_rate))

        # Trim the input buffer
        if self.processed_audio_length > 0:
            self.input_audio_buffer = self.input_audio_buffer[self.processed_audio_length:]
            self.processed_audio_length = 0

        return new_text, logs

    def extract_new_text(self, time_stamped_transcription) -> str:
        if not time_stamped_transcription:
            return ""

        # Convert processed_audio_length to seconds
        processed_time = self.processed_audio_length / self.sampling_rate

        # Get only words from unprocessed audio
        new_words = [
            word["text"]
            for word in time_stamped_transcription
            if word["time_start"] >= processed_time
        ]

        return " ".join(new_words).strip()

    def transcribe(self, source_lang: str) -> List[Dict]:
        start_time = time.time()
        segments, info = self.model.transcribe(
            self.input_audio_buffer,
            language=source_lang,
            task="transcribe",
            beam_size=self.BEAM_SIZE,
            word_timestamps=True,
            condition_on_previous_text=True
        )
        time_stamped_transcription = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > self.NO_SPEECH_PROBABILITY:
                    continue
                time_stamped_transcription.append({
                    "time_start": word.start,
                    "time_end": word.end,
                    "text": word.word
                })

        transcription_time = time.time() - start_time
        log_message = f"Transcription took {transcription_time:.2f} seconds"
        logs = self.add_log(log_message)

        if self.auto_update_period:
            self.update_period = max(0.7 * transcription_time, self.original_update_period)

        return time_stamped_transcription, logs

# Audio preprocessing functions
def resample(y, original_sample_rate, target_sample_rate: int = 16_000):
    return signal.resample(y, int(len(y) * target_sample_rate / original_sample_rate))

def preprocess_audio(y):
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return y

# Initialize the model
model = "large-v2"
whisper_streaming = FasterWhisperStreamingWithTranslation(model_size_or_path=model)

def process_audio(state, new_chunk, source_lang):
    if state is None:
        state = {'transcription': '', 'translation': '', 'logs': []}

    if new_chunk is None:
        return state, state['transcription'], state['translation'], "\n".join(state.get('logs', []))

    original_sample_rate, y = new_chunk
    y = preprocess_audio(y)
    y = resample(y, original_sample_rate)

    new_state, transcribed_text, translated_text, logs = whisper_streaming.listen(y, source_lang, state)
    return new_state, transcribed_text, translated_text, logs

# Create the Gradio interface
demo = gr.Interface(
    process_audio,
    inputs=[
        "state",
        gr.Audio(sources=["microphone"], streaming=True),
        gr.Radio(choices=["en", "es"], value="en", label="Source Language"),
    ],
    outputs=[
        "state",
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Translated Text"),
        gr.Textbox(label="Processing Logs", lines=10)
    ],
    live=True,
    title="Real-time Speech Transcription and Translation",
    description="Speak in English or Spanish to get real-time transcription and translation. The logs window shows processing times."
)

demo.launch(share=True)