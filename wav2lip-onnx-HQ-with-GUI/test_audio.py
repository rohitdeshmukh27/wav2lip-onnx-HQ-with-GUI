import librosa
audio_path = "D:/study/fy project/wav2lip-onnx-HQ-main/Example/test.WAV"
wav = librosa.load(audio_path, sr=16000)[0]
print("Audio loaded successfully, length:", len(wav))