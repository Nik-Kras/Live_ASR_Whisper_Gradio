import gradio as gr
from faster_whisper import WhisperModel
import numpy as np
from scipy import signal
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import nltk
import time

# Download the necessary NLTK data (you only need to do this once)
nltk.download('punkt_tab')


class FasterWhisperStreaming:
    
    BEAM_SIZE = 5
    NO_SPEECH_PROBABILITY = 0.9
    MAX_SENTENCES = 2               # Max number of sentences you update. The rest is kept frozen and untouched

    def __init__(self, model_size_or_path: str ="large-v2", sample_rate: int = 16_000, device: str ="auto", update_period: int = 0.7, lang: str = "en", auto_update_period: bool = False):
        self.model = WhisperModel(model_size_or_path=model_size_or_path, device=device, compute_type="float16")
        self.update_event_buffer = np.array([])         # Buffer to make updates only when this buffer reaches Threshold
        self.input_audio_buffer = np.array([])          # Buffer to the audio for the last sentence that should be updated
        self.original_update_period = update_period     # To remember the original value in case for auto_update_period
        self.update_period = update_period              # Number of seconds to wait until the last sentence should be updated with new audio
        self.auto_update_period = auto_update_period    # Whether to adjust auto_update_period to inference time
        self.sampling_rate = sample_rate                # Audio sample rate (8k, 16k, ..., 44k)
        self.prompt = ""                                # Text that already has been transcribed and should be left untouched
        self.predicted_text = ""                        # Currently predicted text that could be updated later on
        self.lang = lang                                # Speaker's language

    def listen(self, audio_chunk):
        """ Adds audio chunk to the update buffer and calls update (transcription) when update_period waiting time is over """
        self.update_event_buffer = np.concatenate((self.update_event_buffer, audio_chunk))
        if len(self.update_event_buffer) > int(self.update_period * self.sampling_rate):
            self.predicted_text = self.update()
        return self.prompt, self.predicted_text

    def update(self):
        """ Updates transcription with input+new voice chunk. Sends old transcription to prompt if necessary """
        self.input_audio_buffer = np.concatenate((self.input_audio_buffer, self.update_event_buffer))       # Prepare full Input for transcription
        self.update_event_buffer = np.array([])                                                             # Clear temp buffer for events
        time_stamped_transcription = self.transcribe()                                                      # Get transcription
        trimmed_time_stamped_transcription = self.update_prompt(time_stamped_transcription)                 # Keep old transcriptions in prompt if the transcription is too long. It will cut `time_stamped_transcription` if needed
        self.update_input_audio_buffer(trimmed_time_stamped_transcription)                                  # In case transcription was too long -> cut old audio as well
        predicted_text = " ".join([x["text"] for x in trimmed_time_stamped_transcription]).strip()          # Return all predicted words (rest is in self.prompt)
        return predicted_text

    def transcribe(self) -> List[Dict]:
        """ Provides text transcription with timestamps from `self.input_audio_buffer` """
        start = time.time()
        segments, info = self.model.transcribe(self.input_audio_buffer, language=self.lang, initial_prompt=self.prompt, beam_size=self.BEAM_SIZE, word_timestamps=True, condition_on_previous_text=True)
        time_stamped_transcription = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > self.NO_SPEECH_PROBABILITY:
                    continue
                time_stamped_transcription.append({"time_start": word.start, "time_end": word.end, "text": word.word})
                
        end = time.time()
        inference_time = end-start
        print(f"done. It took {round(inference_time,2)} seconds.")
        
        if self.auto_update_period:
            self.update_period = max(0.7 * inference_time, self.original_update_period) 
            
        return time_stamped_transcription

    def update_prompt(self, time_stamped_transcription) -> List[Dict]:
        """ In case of too long transcription - cuts old sentences and stores them in the prompt. Cuts `time_stamped_transcription` respectively """
        sents = self.words_to_sentences(time_stamped_transcription)
        
        # Adding old sentences to prompt
        while len(sents) > self.MAX_SENTENCES:
            last_sentence = sents.pop(0)["text"]
            self.prompt += " " + last_sentence
            self.prompt = self.prompt.strip()
        
        # Filter popped words from time_stamped_transcription
        if sents:
            first_word_time = sents[0]["time_start"]
            time_stamped_transcription = [word for word in time_stamped_transcription if word["time_start"] >= first_word_time]
            
        return time_stamped_transcription

    def update_input_audio_buffer(self, time_stamped_transcription):   
        """ In case `time_stamped_transcription` was cut -> do the same with `input_audio_buffer` """
        if time_stamped_transcription and time_stamped_transcription[0]["time_start"] > 0:
            cut_time = time_stamped_transcription[0]["time_start"]
            cut_samples = int(cut_time * self.sampling_rate)
            self.input_audio_buffer = self.input_audio_buffer[cut_samples:]
            
    def words_to_sentences(self, time_stamped_transcription) -> List[Dict]:
        """  Converts a list of words with timestamps (time_stamped_transcription) into a list of sentences with timestamps """
         
        text = " ".join(word["text"] for word in time_stamped_transcription)
        sentences = sent_tokenize(text) 
                
        out = []
        word_index = 0
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_words = sentence.split()
            sentence_start = time_stamped_transcription[word_index]["time_start"]
            
            while word_index < len(time_stamped_transcription) and len(sentence_words) > 0:
                if time_stamped_transcription[word_index]["text"].strip() == sentence_words[0]:
                    sentence_words.pop(0)
                    if len(sentence_words) == 0:
                        sentence_end = time_stamped_transcription[word_index]["time_end"]
                        out.append({"time_start": sentence_start, "time_end": sentence_end, "text": sentence})
                word_index += 1
        
        return out


def resample(y, original_sample_rate, target_sample_rate: int = 16_000):
    return signal.resample(y, int(len(y) * target_sample_rate / original_sample_rate))


def preprocess_audio(y):
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return y


# See available models here: https://huggingface.co/Systran
RECOMMENDED_MODELS = [
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v2",
    "distil-large-v3"
]

model = RECOMMENDED_MODELS[0]
whisper_streaming_model = FasterWhisperStreaming(model_size_or_path=model)


def transcribe(full_audio, new_chunk):

    original_sample_rate, y = new_chunk
    y = preprocess_audio(y)
    y = resample(y, original_sample_rate)

    certain_text, uncertain_text = whisper_streaming_model.listen(y)
    
    text = certain_text + "***" + uncertain_text
    
    return full_audio, text

demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch(share=True)
