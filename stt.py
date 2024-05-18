from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np
import time
import queue

class STT:
  def __init__(self):
    model = "openai/whisper-base.en"
    self.processor = WhisperProcessor.from_pretrained(model, cache_dir="models")
    self.model = WhisperForConditionalGeneration.from_pretrained(model, cache_dir="models")
    self.model.config.forced_decoder_ids = None

  def duration_transcribe(self, duration_seconds=3):
    fs = 16000
    record = sd.rec(int(duration_seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    sample = {"array": np.squeeze(record), "sampling_rate": fs}

    return self.transcribe_audio(sample)

  def dynamic_transcribe(
    self, max_duration_seconds=30, silence_duration=2
  ):
    fs = 16000
    record = queue.Queue()
    silence_start = None
    min_volume = float("inf")
    max_volume = float(0)
    callback_aborted = False
    start_time = time.time()

    def callback(in_data, frames, time_info, status):
      nonlocal callback_aborted
      nonlocal silence_start
      nonlocal min_volume
      nonlocal max_volume
      nonlocal record

      if status:
        print(status)

      assert len(in_data) == frames

      volume_norm = np.linalg.norm(in_data) * 10
      min_volume = min(min_volume, volume_norm)
      max_volume = max(max_volume, volume_norm)
      loud_enough = volume_norm >= (min_volume + max_volume) / 2
      now = time.time()

      if loud_enough and silence_start is not None:
        silence_start = None
      elif not loud_enough and silence_start is None:
        silence_start = now

      if silence_start is not None and now - silence_start > silence_duration:
        callback_aborted = True
        raise sd.CallbackAbort
      elif now - start_time > max_duration_seconds:
        callback_aborted = True
        raise sd.CallbackAbort
      
      record.put(in_data.copy())

    with sd.InputStream(callback=callback, samplerate=fs, channels=1):
      while not callback_aborted:
        sd.sleep(500)

    all_data = []
    while not record.empty():
        all_data.append(record.get())

    sample = {"array": np.squeeze(np.concatenate(all_data)), "sampling_rate": fs}

    return self.transcribe_audio(sample)

  def transcribe_audio(self, audio=None):
    if audio is None:
      return ""

    input_features = self.processor(
      audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    predicted_ids = self.model.generate(input_features)
    transcription = self.processor.batch_decode(
      predicted_ids, skip_special_tokens=True
    )

    return transcription[0]


if __name__ == "__main__":
  stt = STT()
  print("Speak now...")
  transcription = stt.dynamic_transcribe()
  print("Recording finished.")
  print("Transcription:", transcription)
