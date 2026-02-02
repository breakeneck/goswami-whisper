import torch
from qwen_asr import Qwen3ASRModel

import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Qwen3 ASR on one or more audio files.")
    parser.add_argument("audio", nargs="*", help="Path(s) to audio file(s) to transcribe.")
    args = parser.parse_args()
    audio_files = args.audio
    if not audio_files:
        print("No audio files provided; running with empty audio list.", file=sys.stderr)

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    )

    results = model.transcribe(
        audio=audio_files[0] if audio_files else "",
        language="Russian", # set "English" to force the language
    )

    print(results[0].language)
    print(results[0].text)