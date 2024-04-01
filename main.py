import whisper
import os
import json
import nltk
import requests
import threading
import time
import logging
from datetime import datetime

nltk.download("punkt")

STREAM_ID = 1189
CHUNK_PER_SECOND = 1024 * 2
RAW_AUDIO_PATH = "./files/raw_audio"
TRANSCRIBED_PATH = "./files/transcribed"
THREATS_PATH = "./files/threats"
THREAT_LIBRARY = "threat_library.json"

logging.basicConfig(level=logging.INFO)
stop_event = threading.Event()

def data_gathering(url, stream_id, chunk_per_second):
    while not stop_event.is_set():
        logging.info(f"data_gathering stop_event {stop_event}")
        response = requests.get(url, stream=True)
        for chunk in response.iter_content(chunk_size=chunk_per_second * 60):
            formatted_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            file_path = os.path.join(RAW_AUDIO_PATH, f"{stream_id}_{formatted_date}.mp3")

            with open(file_path, "wb") as f:
                f.write(chunk)
            logging.info(f"Saved audio chunk at {file_path}")


def transcription(model_name):
    model = whisper.load_model(model_name)
    options = whisper.DecodingOptions().__dict__.copy()
    options['no_speech_threshold'] = 0.3

    while not stop_event.is_set():
        logging.info(f"transcription stop_event {stop_event}")
        for file in os.listdir(RAW_AUDIO_PATH):
            audio_path = os.path.join(RAW_AUDIO_PATH, file)
            output_path = os.path.join(TRANSCRIBED_PATH, f"{file}.txt")

            result = whisper.transcribe(model, audio_path)
            if result and result["text"]:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                logging.info(f"Saved transcription at {output_path}")
            else:
                os.remove(audio_path)
                
        time.sleep(10)


def threat_detection():
    with open(THREAT_LIBRARY) as f:
        threats_keywords = json.load(f)
        
    while not stop_event.is_set():
        logging.info(f"threat_detection stop_event {stop_event}")
        for file in os.listdir(TRANSCRIBED_PATH):
            transcribed_path = os.path.join(TRANSCRIBED_PATH, file)
            threats_file_path = os.path.join(THREATS_PATH, f"{file}.txt")

            found_threats = set()
            with open(transcribed_path) as f:
                text = f.read()
                tokens = nltk.word_tokenize(text)
                for token in tokens:
                    for category in threats_keywords:
                        if token in category["keywords"]:
                            found_threats.add(token)

            if found_threats:
                with open(threats_file_path, "w") as f:
                    f.write("\n".join(found_threats))
                logging.info(f"Saved found threats at {threats_file_path}")
            # else:
            #     os.remove(transcribed_path)
        time.sleep(10)


if __name__ == "__main__":
    try:
        data_gathering_thread = threading.Thread(
            target=data_gathering,
            args=(
                "https://broadcastify.cdnstream1.com/1189",
                STREAM_ID,
                CHUNK_PER_SECOND,
            ),
        )
        transcription_thread = threading.Thread(target=transcription, args=("medium.en"))
        threat_detection_thread = threading.Thread(target=threat_detection, args=())

        data_gathering_thread.start()
        transcription_thread.start()
        threat_detection_thread.start()

        while not stop_event.is_set():
            data_gathering_thread.join(1)
            transcription_thread.join(1)
            threat_detection_thread.join(1)

    except KeyboardInterrupt:
        logging.info("Exiting...")
        stop_event.set()
        data_gathering_thread.join()
        transcription_thread.join()
        threat_detection_thread.join()