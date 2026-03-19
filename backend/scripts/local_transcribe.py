import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: local_transcribe.py <audio_path> [language] [model]", file=sys.stderr)
        return 1

    audio_path = Path(sys.argv[1])
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "base"

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        print(f"Failed to import faster_whisper: {exc}", file=sys.stderr)
        return 1

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(audio_path),
            language=None if language == "auto" else language,
            vad_filter=False,
            beam_size=5,
        )

        text_parts = []
        for segment in segments:
            value = (segment.text or "").strip()
            if value:
                text_parts.append(value)

        text = " ".join(text_parts).strip()
        payload = {
            "text": text,
            "language": getattr(info, "language", language),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Local whisper transcription failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
