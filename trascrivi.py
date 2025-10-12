import whisper
from pathlib import Path

# <<< CAMBIA QUI con il percorso del tuo MP3 >>>
audio_path = Path(r"C:\Users\emman\Desktop\Projects\BotInvestimento\trading.mp3")

model = whisper.load_model("small")  # "base" è più leggero, "small" più accurato
result = model.transcribe(str(audio_path), language="en")  # o lascia auto-detect

text = result.get("text", "").strip()
print(text)

# Salva anche su file
out_path = audio_path.with_suffix(".txt")
out_path.write_text(text, encoding="utf-8")
print(f"\nTrascrizione salvata in: {out_path}")
