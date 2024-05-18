from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch

class Trigger:
  def __init__(self, wake_word="sheila", stop_word="stop", prob_threshold=0.5, chunk_length_s=2.0, stream_chunk_s=0.25):
    self.wake_word = wake_word
    self.stop_word = stop_word
    self.prob_threshold = prob_threshold
    self.chunk_length_s = chunk_length_s
    self.stream_chunk_s = stream_chunk_s

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.classifier = pipeline(
      "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device,
      model_kwargs={"cache_dir": "models"}
    )

    self.sampling_rate = self.classifier.feature_extractor.sampling_rate

    if wake_word not in self.classifier.model.config.label2id.keys():
      raise ValueError(
        f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {self.classifier.model.config.label2id.keys()}."
      )
    if stop_word not in self.classifier.model.config.label2id.keys():
      raise ValueError(
        f"Wake word {stop_word} not in set of valid class labels, pick a wake word in the set {self.classifier.model.config.label2id.keys()}."
      )

  def wait(self):
    mic = ffmpeg_microphone_live(
      sampling_rate=self.sampling_rate,
      chunk_length_s=self.chunk_length_s,
      stream_chunk_s=self.stream_chunk_s,
    )

    for prediction in self.classifier(mic):
      prediction = prediction[0]
      if prediction["label"] == self.wake_word:
        if prediction["score"] > self.prob_threshold:
          return "wakeup"
      if prediction["label"] == self.stop_word:
        if prediction["score"] > self.prob_threshold:
          return "exit"
      
if __name__ == "__main__":
  trigger = Trigger()

  result = trigger.wait()

  if result == "wakeup":
    print("Wakeup word detected!")
  elif result == "exit":
    print("Exit word detected!")
