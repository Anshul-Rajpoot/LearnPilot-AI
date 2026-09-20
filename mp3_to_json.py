import whisper
import json
import os

# Load the Whisper model once to reuse it for all audio files
model = whisper.load_model("large-v2")

# Get all audio files from the input directory
audios = os.listdir("audios")

for audio in audios:
    # Process only files following the "<number>_<title>.ext" naming convention
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]

        print(number, title)

        # Transcribe Hindi speech and translate it into English
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        # Store each transcript segment with its metadata
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        # Save both segmented transcript and complete translated text
        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        # Export transcript as a JSON file
        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)
