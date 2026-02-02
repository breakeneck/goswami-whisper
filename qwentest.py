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

    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=4096, # Maximum number of tokens to generate. Set a larger value for long audio input.
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
        ),
    )

    results = model.transcribe(
        audio=audio_files,
        language=["Russian"],
        return_time_stamps=False,
    )

    for r in results:
        print(r.language, r.text, r.time_stamps[0])