import os
import json
import soundfile as sf
import argparse
import glob
from tqdm import tqdm
from lhotse.shar.readers.lazy import LazySharIterator

# CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo_s2s_24.08zhc.sqsh
# export HF_DATASETS_CACHE=/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE/
# enroot start --mount /lustre:/lustre --mount /home:/home --mount $HF_DATASETS_CACHE:$HF_DATASETS_CACHE --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE ${CONTAINER}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/nvolve_chatqa/")
    parser.add_argument("--output_audio_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/nvolve_chatqa/shar_duplex/recording")
    parser.add_argument("--output_manifest_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/nvolve_chatqa/shar_duplex")
    return parser.parse_args()

def replace_cuts_name(cuts_name, new_name=None, new_root=None):
    cuts_name = [name.replace("jsonl.gz", "tar") for name in cuts_name]
    if new_root is not None:
        cuts_name = [os.path.join(new_root, os.path.basename(name)) for name in cuts_name]
    if new_name is not None:
        cuts_name = [name.replace("cuts", new_name) for name in cuts_name]
    return cuts_name

def main():
    args = parse_args()
    
    in_shar_dir = args.in_shar_dir
    output_audio_dir = args.output_audio_dir
    output_manifest_dir = args.output_manifest_dir

    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_manifest_dir, exist_ok=True)
    
    cuts_files = sorted(glob.glob(f"{in_shar_dir}/cuts.*.jsonl.gz"))
    
    manifest_path = os.path.join(output_manifest_dir, 'nfa_manifest.jsonl')
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for cuts_file in cuts_files:
            cuts = LazySharIterator(
                {
                    "cuts": [cuts_file],
                    "recording": replace_cuts_name([cuts_file], new_name="recording"),
                    "target_audio": replace_cuts_name([cuts_file], new_name="target_audio"),
                }
            )

            for j, cut in enumerate(tqdm(cuts, desc=f"Processing {os.path.basename(cuts_file)}")):
                if not cut.supervisions or cut.recording is None:
                    continue
                
                for i in range(0, len(cut.supervisions), 2):
                    question = cut.supervisions[i].text
                    # print(f"supervision[{i}].text:", question)
                    samples = cut.recording.load_audio()
                    sample_rate = cut.recording.sampling_rate
                    
                    start = int(cut.supervisions[i].start * sample_rate)
                    duration = int(cut.supervisions[i].duration * sample_rate)
                    end = start + duration
                    
                    id = f'{cut.id}_user{i//2}'
                    output_audio_path = os.path.join(output_audio_dir, f"{id}.wav")

                    sf.write(output_audio_path, samples[:, start:end].T, sample_rate)
                    cut_metadata = {
                        "id": f'{id}',
                        "text": question,
                        "audio_filepath": output_audio_path
                    }
                    mf.write(json.dumps(cut_metadata, ensure_ascii=False) + "\n")
    
    print(f"Processing complete. All manifests saved in {output_manifest_dir}")

if __name__ == "__main__":
    main()
