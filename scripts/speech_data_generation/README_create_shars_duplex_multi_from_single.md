# create_shars_duplex_multi_from_single.py

Convert single-turn conversation manifests into multi-turn duplex lhotse shar files.

## What it does

This script takes a JSONL manifest of single-turn conversations (each with one user turn and one agent turn) and combines pairs of them into multi-turn conversations. It then exports the result as [lhotse shar](https://lhotse.readthedocs.io/en/latest/cuts.html#shar) files with two audio streams:

- **recording** (user audio channel): user speech in the active regions, silence during agent speech
- **target_audio** (agent audio channel): agent speech in the active regions, silence during user speech

### Multi-turn construction

Each output sample is a 2-round conversation created by pairing conversation `j` with conversation `N - j - 1` (where N is the total number of valid conversations). This means:

```
Output sample j:
  Round 1: user turn from conversation j  -> agent turn from conversation j
  Round 2: user turn from conversation N-j-1 -> agent turn from conversation N-j-1
```

A configurable silence gap (default 0.32s) is inserted between user speech and the following agent turn.

## Input manifest format

A JSONL file where each line is a JSON object with a `conversations` field. Each conversation has exactly 2 entries: a user turn and an agent turn.

```jsonl
{"conversations": [{"value": "/path/to/user_audio.wav", "instruction": "What is the capital of France?", "from": "user"}, {"value": "/path/to/agent_audio.wav", "transcript": "The capital of France is Paris.", "from": "agent"}]}
{"conversations": [{"value": "/path/to/user_audio2.wav", "instruction": "Tell me about gravity.", "from": "user"}, {"value": "/path/to/agent_audio2.wav", "transcript": "Gravity is a fundamental force...", "from": "agent"}]}
```

### Field descriptions

**User turn** (conversations[0]):
| Field | Required | Description |
|---|---|---|
| `value` | Yes | Path to user audio wav file |
| `instruction` | No | User text (used as supervision text). Defaults to `""` if absent |
| `from` | Yes | Speaker label (e.g. `"user"`) |
| `lang` | No | Language code. Defaults to `"EN"` |

**Agent turn** (conversations[1]):
| Field | Required | Description |
|---|---|---|
| `value` | Yes | Path to agent audio wav file |
| `transcript` | Yes | Agent response text. Entries where transcript is `"I could not find the answer in the audio."` are skipped |
| `from` | Yes | Speaker label (e.g. `"agent"`) |
| `lang` | No | Language code. Defaults to `"EN"` |

## Usage

```bash
python create_shars_duplex_multi_from_single.py \
    --manifest /path/to/input_manifest.jsonl \
    --out_shar_dir /path/to/output/shars \
    --num_shard 10 \
    --dataset_name my_dataset
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--manifest` | str | (see script) | Path to input JSONL manifest |
| `--out_shar_dir` | str | (see script) | Output directory for lhotse shar files |
| `--num_shard` | int | 10 | Number of shards to split the output into |
| `--dataset_name` | str | `"squadv2"` | Dataset name (for logging purposes) |

## Output

The output directory will contain lhotse shar files:

```
out_shar_dir/
  cuts-000000.jsonl.gz
  recording-000000.flac.tar
  target_audio-000000.flac.tar
  ...
```

Each cut in the shar has:
- **recording**: the user audio channel (FLAC)
- **target_audio**: the agent audio channel (FLAC)
- **supervisions**: list of `SupervisionSegment` entries with `start`, `duration`, `text`, and `speaker` for each turn

## Dependencies

- NeMo (`nemo.utils`)
- lhotse
- librosa
- numpy
- soundfile
- torch
- tqdm
- matplotlib

## Notes

- Audio files referenced in the manifest must exist and be readable.
- The script replaces `"fs7"` with `"fsw"` in audio paths (legacy path migration) -- you may want to remove this if not applicable to your setup (line 54).
- The silence between turns is 0.32 seconds by default (hardcoded in `turn_silence_sec` parameter of `create_shar_from_manifest`).
- Conversations where the agent transcript is `"I could not find the answer in the audio."` are filtered out.
