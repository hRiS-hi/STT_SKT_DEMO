from transformers import AutoModel
import torch, torchaudio
import google.generativeai as genai
import os


def load_audio(file_path, target_sr=16000):
    """Robust audio loading function with multiple backend attempts"""
    # Try different torchaudio backends
    backends = ['soundfile', 'sox_io']
    for backend in backends:
        try:
            torchaudio.set_audio_backend(backend)
            wav, sr = torchaudio.load(file_path)
            
            # Convert to mono if necessary
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                wav = resampler(wav)
            
            return wav, target_sr
        except Exception as e:
            print(f"Backend {backend} failed: {str(e)}")
            continue
    
    # Fallback to librosa if all else fails
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        wav = torch.from_numpy(y).unsqueeze(0)
        return wav, sr
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {str(e)}")

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# Load and process audio
wav, sr = load_audio("sample_audio_infer_ready.wav")

# Perform ASR with CTC decoding
transcription_ctc = model(wav, "sa", "ctc")
print("CTC Transcription:", transcription_ctc)

# Perform ASR with RNNT decoding
transcription_rnnt = model(wav, "sa", "rnnt")
print("RNNT Transcription:", transcription_rnnt)


# Save transcriptions to a text file
output_file = "transcription_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"CTC Transcription: {transcription_ctc}\n")
    f.write(f"RNNT Transcription: {transcription_rnnt}\n")

print(f"\nTranscriptions have been saved to {output_file}")