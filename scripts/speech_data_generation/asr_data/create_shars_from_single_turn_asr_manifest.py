import copy
import os
from io import BytesIO
import numpy as np
from speech_data_generation.utils import save_audio
from speech_data_generation.data.recording import Recording
from speech_data_generation.data.supervision_segment import SupervisionSegment

def create_shars_from_single_turn_asr_manifest(entries, sampling_rate, silence_padding):
    new_cuts = []
    for entry in entries:
        # Add supervision
        new_cut = copy.deepcopy(cut)
        cut_id = os.path.basename(cut.id).split(".")[0]
        new_cut.id = cut_id
        new_cut.supervisions = [
            SupervisionSegment(
                id=cut_id,
                recording_id=cut_id,
                start=0,
                duration=0,  # will update below
                text=entry["text"],
                speaker="user",
                language="EN",
            )
        ]

        # Create zero-filled agent audio with same length as user audio
        user_audio = cut.recording.load_audio()
        user_audio = np.concatenate([user_audio, silence_padding], axis=1)
        agent_audio = np.zeros_like(user_audio)

        # Set duration based on actual audio length
        new_cut.supervisions[0].duration = agent_audio.shape[1] / sampling_rate
        new_cut.duration = agent_audio.shape[1] / sampling_rate
        new_cut.start = 0.0
    
        user_stream = BytesIO()
        agent_stream = BytesIO()
        save_audio(dest=user_stream, src=user_audio, sampling_rate=sampling_rate, format="wav")
        save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sampling_rate, format="wav")
        user_stream.seek(0)
        agent_stream.seek(0)
        new_cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_id}_user")
        new_cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut_id}_agent")
        new_cuts.append(new_cut)
    return new_cuts 