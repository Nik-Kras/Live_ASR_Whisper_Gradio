from faster_whisper import WhisperModel

model = WhisperModel("large-v2")

segments, info = model.transcribe("you-were-my-brother-anakin-i-loved-you.wav", language="en", condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
