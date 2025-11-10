from transformers import AutoModel
import torch
import torchaudio
import os
import sounddevice as sd
import soundfile as sf
import numpy as np




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
            # print(f"Backend {backend} failed: {str(e)}") # Optional: comment out to reduce noise
            continue
    
    # Fallback to librosa if all else fails
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        wav = torch.from_numpy(y).unsqueeze(0)
        return wav, sr
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {str(e)}")

def record_from_microphone(filename="temp_recording.wav", fs=16000):
    """
    Records audio from the default microphone until the user presses Enter.
    Saves the recording to a temporary WAV file.
    """
    print("\n" + "="*40)
    print("🎤  PREPARING TO RECORD")
    print("="*40)
    input("Press Enter to START recording...")
    
    print("\n🔴 RECORDING... (Press Enter again to STOP)")
    
    # We record in chunks and store them in a list
    recorded_chunks = []
    recording = True

    # Use a callback to capture audio without blocking the main thread completely
    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        if recording:
            recorded_chunks.append(indata.copy())

    # Start the recording stream
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        # This input() blocks the main thread while the callback continues to record in the background
        input() 
        recording = False # Stop appending new data in callback

    print("⏹️  Recording stopped. Processing...")

    # Concatenate all chunks into one numpy array
    if not recorded_chunks:
        print("Warning: No audio recorded.")
        return None
        
    myrecording = np.concatenate(recorded_chunks, axis=0)
    
    # Save as a standard WAV file for the rest of the pipeline to use
    sf.write(filename, myrecording, fs)
    print(f"Audio saved to {filename}")
    return filename

# =========================================
# MAIN PROGRAM FLOW
# =========================================

print("Loading AI Model... (this may take a moment)")
# Load the model ONCE at startup
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
print("✅ Model loaded successfully.\n")

while True:
    # Main interactive loop
    user_choice = input("\nType 'r' to record new audio, or 'q' to quit: ").lower().strip()
    
    if user_choice == 'q':
        print("Exiting program.")
        break
    elif user_choice == 'r':
        # 1. Record Audio
        audio_filename = record_from_microphone()
        
        if audio_filename and os.path.exists(audio_filename):
            print("\n🧠 Transcribing...")
            try:
                # 2. Load and Process (using your robust function)
                wav, sr = load_audio(audio_filename)

                # 3. Perform Inference
                # You can change "sa" (Sanskrit) to "hi" (Hindi) or others if needed
                transcription_ctc = model(wav, "sa", "ctc")
                transcription_rnnt = model(wav, "sa", "rnnt")
                
                print("\n--- Transcription Results ---")
                
                print(transcription_rnnt)

                print("✅ Transcription Complete:")

                # 5. Save to file (append mode 'a' is better for a running log)
                output_file = "transcription_log.txt"
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n--- New Recording ---\n")
                    f.write(f"CTC: {transcription_ctc}\n")
                    f.write(f"RNNT: {transcription_rnnt}\n")
                print(f"Saved to {output_file}")

            except Exception as e:
                print(f"❌ An error occurred during transcription: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
        else:
            print("Recording failed or was empty.")
    else:
        print("Invalid choice. Please type 'r' or 'q'.")