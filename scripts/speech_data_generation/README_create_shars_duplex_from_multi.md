# create_shars_duplex_from_multi.py

Convert multi-turn conversation manifests into duplex lhotse shar files.

## What it does

This script takes a JSONL manifest where each entry already contains a multi-turn conversation (multiple user/agent turn pairs) and exports it as [lhotse shar](https://lhotse.readthedocs.io/en/latest/cuts.html#shar) files with two audio streams:

- **recording** (user audio channel): user speech in the active regions, silence during agent speech
- **target_audio** (agent audio channel): agent speech in the active regions, silence during user speech

Unlike `create_shars_duplex_multi_from_single.py` (which pairs single-turn conversations to construct multi-turn samples), this script assumes each manifest entry is already a complete multi-turn conversation.

### Audio construction

For each conversation, the script iterates over the turn pairs and concatenates them sequentially:

```
Time ->
User channel:   [user_audio_1] [silence........] [user_audio_2] [silence........]
Agent channel:  [silence.....] [agent_audio_1..] [silence.....] [agent_audio_2..]
```

The silence regions are zero-filled and match the duration of the corresponding speaker's audio from the other channel. No additional silence gap is inserted between turns.

## Input manifest format

A JSONL file where each line is a JSON object with a `conversations` field. Each conversation has an even number of entries, alternating between user and agent turns.

```jsonl
{"conversations": [{"value": "/path/to/user1.wav", "instruction": "What is AI?", "from": "user"}, {"value": "/path/to/agent1.wav", "transcript": "AI stands for...", "from": "agent"}, {"value": "/path/to/user2.wav", "instruction": "Tell me more.", "from": "user"}, {"value": "/path/to/agent2.wav", "transcript": "Sure, AI can...", "from": "agent"}]}
```

### Field descriptions

**User turn** (conversations[0], conversations[2], ...):
| Field | Required | Description |
|---|---|---|
| `value` | Yes | Path to user audio wav file |
| `instruction` | No | User text (used as supervision text). Defaults to `""` if absent |
| `from` | Yes | Speaker label (e.g. `"user"`) |
| `lang` | No | Language code. Defaults to `"EN"` |

**Agent turn** (conversations[1], conversations[3], ...):
| Field | Required | Description |
|---|---|---|
| `value` | Yes | Path to agent audio wav file |
| `transcript` | Yes | Agent response text |
| `from` | Yes | Speaker label (e.g. `"agent"`) |
| `lang` | No | Language code. Defaults to `"EN"` |

## Usage

```bash
python create_shars_duplex_from_multi.py \
    --manifest /path/to/input_manifest.jsonl \
    --out_shar_dir /path/to/output/shars \
    --num_shard 10
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--manifest` | str | (see script) | Path to input JSONL manifest |
| `--out_shar_dir` | str | (see script) | Output directory for lhotse shar files |
| `--num_shard` | int | 10 | Number of shards to split the output into |

## Output

The output directory will contain lhotse shar files:

```
out_shar_dir/
  cuts-000000.jsonl.gz
  recording-000000.wav.tar
  target_audio-000000.wav.tar
  ...
```

Each cut in the shar has:
- **recording**: the user audio channel (WAV)
- **target_audio**: the agent audio channel (WAV)
- **supervisions**: list of `SupervisionSegment` entries with `start`, `duration`, `text`, and `speaker` for each turn

## Comparison with create_shars_duplex_multi_from_single.py

| Feature | `from_multi` (this script) | `from_single` |
|---|---|---|
| Input format | Already multi-turn conversations | Single-turn conversations |
| Multi-turn construction | Uses turns as-is from manifest | Pairs conversation j with N-j-1 |
| Silence between turns | None (zero-filled to match other channel) | Configurable (default 0.32s) |
| Audio format | WAV | FLAC |
| Temp files | Uses `/tmp/` for intermediate audio | No temp files |

## Dependencies

- lhotse
- librosa
- numpy
- soundfile
- torch
- tqdm
- matplotlib

## Notes

- Audio files referenced in the manifest must exist and be readable.
- The script replaces `"fs7"` with `"fsw"` in audio paths (legacy path migration) -- you may want to remove this if not applicable to your setup.
- Intermediate audio files are written to `/tmp/` during processing. Ensure sufficient disk space.
- The number of conversation entries must be even (alternating user/agent turns).
- User audio is resampled to match the agent audio's sample rate.
