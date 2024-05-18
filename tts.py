from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import sounddevice as sd

class TTS:
    def __init__(self) -> None:
        self.models, self.cfg, self.task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            cache_dir="models",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        self.model = self.models[0]
        TTSHubInterface.update_cfg_with_data_cfg(self.cfg, self.task.data_cfg)
        self.generator = self.task.build_generator([self.model], self.cfg)

    def synthesize(self, text: str):
        sample = TTSHubInterface.get_model_input(self.task, text)
        wav, rate = TTSHubInterface.get_prediction(self.task, self.model, self.generator, sample)
        sd.play(wav, samplerate=rate)
        sd.wait()

if __name__ == "__main__":
    tts = TTS()
    tts.synthesize("Hello World!")
