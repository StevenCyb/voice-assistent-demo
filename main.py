from trigger import Trigger
from stt import STT
from chat import Chat
from tts import TTS

trigger = Trigger()
stt = STT()
chat = Chat()
tts = TTS()
while True:
  print("Waiting for command...")
  result = trigger.wait()

  if result == "wakeup":
    print("Wakeup word detected (sheila)")
    print("Recording started...")
    command = stt.dynamic_transcribe()
    print("Recording finished.")

    print(f"Command: {command}")
    response = chat.ask(command)
    if response == "":
      response = "I'm sorry, I didn't understand that."
    print(f"Response: {response}")

    print("Speaking response...")
    tts.synthesize(response)
  elif result == "exit":
    print("Exit word detected!")
    break