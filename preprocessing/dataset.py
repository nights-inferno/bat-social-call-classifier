"""Audio Dataset and Preprocessing Pipeline for Pipistrelle Bat Recordings

"""
import math
import random
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import pandas as pd
import numpy as np

import soundfile as sf
import torchaudio
import torchaudio.transforms as AT
import torchaudio.functional as AF
from scipy.signal import butter, sosfilt

class BatAudioPipeline(torch.nn.Module):
    def __init__(
        self, 
        target_sr=16000, 
        expansion_factor=10, 
        window_sec=10, 
        overlap=0.5,
        filter_echo : bool = False,
    ) -> None :
        
        super().__init__()
        self.target_sr = target_sr
        self.expansion_factor = expansion_factor
        
        # Windowing parameters
        self.win_samples = window_sec * target_sr
        self.hop_samples = int(self.win_samples * (1.0 - overlap))
        self.filter_echo = filter_echo
        self.orig_sr = None

    def load(self, file_path : str) -> torch.Tensor:
        """Loads audio using soundfile and applies 10x time expansion logic."""
        try:
            # Load using soundfile
            data, orig_sr = sf.read(file_path)
            self.orig_sr = orig_sr

            # Convert to torch tensor [channels, time]
            # soundfile returns [time, channels] for stereo, so we transpose
            audio = torch.from_numpy(data).float()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0) # Add channel dim for mono
            else:
                audio = audio.transpose(0, 1) # [T, C] -> [C, T]

            # Convert stereo to mono if necessary
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            return audio

        except Exception as e:
            print(f"\n[Warning] Skipping {file_path}: {e}")
            # Return 1 second of silence as a fallback to prevent NoneType errors
            return torch.zeros((1, self.target_sr))
    
    def apply_time_expansion(self,audio: torch.Tensor) -> torch.Tensor:
        # Time Expansion logic
        virtual_sr = self.orig_sr // self.expansion_factor
        # Resample to target sample rate
        if virtual_sr != self.target_sr:
            audio = AF.resample(audio, orig_freq=virtual_sr, new_freq=self.target_sr)
        return audio

    def apply_bandpass(self, audio : torch.Tensor) -> torch.Tensor:
        """
        Applies a high-order Butterworth filter using SciPy for a 
        sharp cutoff before converting to PyTorch. Filters out low noise
        and 
        """
        # 1. Ensure orig_sr is an integer
        fs = float(self.orig_sr)
        # Convert torch tensor back to numpy temporarily for SciPy
        audio_np = audio.cpu().numpy()

        # 1.  8th-order Highpass at 15 kHz (48 dB/octave roll-off)
        sos_hp = butter(N=8, Wn=15000, btype='highpass', fs=fs, output='sos')
        audio_np = sosfilt(sos_hp, audio_np)

        # 2. Optional:  Bandreject for Echolocation Echoes
        if self.filter_echo:
            fmin, fmax = 40000, 75000
            nyquist = fs / 2
            if fmin >= nyquist:
                pass 
            else:
                # Clamp fmax to just below the Nyquist limit
                fmax = min(fmax, nyquist - 1) 

                sos_br = butter(N=8, Wn=[fmin, fmax], btype='bandstop', fs=fs, output='sos')
                audio_np = sosfilt(sos_br, audio_np)

        return torch.from_numpy(audio_np).float()

    def window_audio(self, audio : torch.Tensor) -> torch.Tensor:
        """Cuts the 1D audio tensor into overlapping windows."""
        # Shape goes from [1, Total_Samples] -> [1, Num_Windows, Window_Samples]
        if audio.shape[1] < self.win_samples:
            # Pad with zeros if the file is shorter than window size
            pad_amount = self.win_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
            
        windows = audio.unfold(-1, self.win_samples, self.hop_samples)
        # Remove channel dimension [Num_Windows, Window_Samples] 
        windows = windows.squeeze(0)
        return windows
    
    def time_shift(self, audio: torch.Tensor, shift_ratio=0.5) -> torch.Tensor:
        max_shift = int(shift_ratio * self.win_samples)
        shift = random.randint(0, max_shift)
    
        if shift > 0:
            # Deletes the first 'shift' samples along the last axis
            # and returns the rest of the window
            return audio[..., shift:]
        
        return audio

    def forward(self, file_path : str,timeshift : bool = False) -> torch.Tensor:

        # 0. Load & Time Expand
        audio = self.load(file_path)
        
        # 1. Bandpass Filter
        audio = self.apply_bandpass(audio)

        #2. Time expand
        audio = self.apply_time_expansion(audio)

        # 3. Data augmentation
        if timeshift :
            audio = self.time_shift(audio)

        # 4. Cut into Windows
        windows = self.window_audio(audio)
        
        return windows
    



class AugmentationPipeline:
    def __init__(
        self, 
        online_augment=None,
        resample_augment=None
    ) -> None:

        # Format: [Enabled (bool), "aug_name", ...]
        self.online_augment = online_augment if online_augment else [False]
        self.resample_augment = resample_augment if resample_augment else [False]

    def get_ir_per_label(self, y : np.ndarray) -> np.ndarray:
        """Calculates the Imbalance Ratio per Label (IRLBL)."""
        counts = np.sum(y, axis=0)
        max_count = np.max(counts)
        # Avoid division by zero for labels with 0 occurrences
        ir_per_label = max_count / (counts + 1e-9)
        return ir_per_label
        
    def iterative_oversample(self, X, y, target_percentage=0.2,random_state = 42):
        """
        Randomly duplicates samples containing minority labels 
        until the distribution balances out.
        """
        rng = np.random.default_rng(random_state)
        # Convert to numpy for indexing if they aren't already
        X_resampled = list(X)
        y_resampled = list(y)

        current_counts = np.sum(y_resampled, axis=0)
        max_label_count = np.max(current_counts)

        # We want every label to at least reach a certain percentage 
        # of the majority label's count. 
        target_count = max_label_count * target_percentage

        # Indices of samples grouped by label for quick access
        label_to_indices = {i: np.where(y[:, i] == 1)[0] for i in range(y.shape[1])}

        # Keep adding samples until all labels meet the target
        balancing = True
        while balancing:
            ir_labels = self.get_ir_per_label(np.array(y_resampled))

            # Find labels that are still below target_count
            underrepresented_labels = np.where(np.sum(y_resampled, axis=0) < target_count)[0]

            if len(underrepresented_labels) == 0:
                balancing = False
                break

            # Focus on the most imbalanced label currently
            worst_label = underrepresented_labels[np.argmax(ir_labels[underrepresented_labels])]

            # Pick a random sample that contains this label
            possible_indices = label_to_indices[worst_label]
            if len(possible_indices) == 0:
                continue # Should not happen if label exists

            idx_to_clone = rng.choice(possible_indices)

            # Duplicate the sample
            y_resampled.append(y[idx_to_clone])
            X_resampled.append(X[idx_to_clone])

            # (Optional) Stop if we exceed a certain size to prevent infinite loops
            if len(y_resampled) > len(y) * 3:
                print("Reached safety limit (3x original size). Stopping.")
                break

        print(f"Final counts: {np.sum(y_resampled, axis=0).astype(int)}")
        return np.array(X_resampled), np.array(y_resampled)
    
    def augment_sample(self, windows, aug_list, noise_files, pipeline_ref):
        """Applies a list of string-named augmentations to an audio tensor."""
        # The first element is the boolean toggle
        if not aug_list[0]: 
            return windows

        augmented_windows = windows.clone()
        
        for aug in aug_list[1:]:
            if aug == "time_shift":
                # Maximum safe shift of 25% of the window
                window_len = windows.shape[-1]
                max_shift = int(0.25 * window_len) 
                shift = random.randint(-max_shift, max_shift)
                
                if shift != 0:
                    # Construct clean padding to prevent wrap-around artifacts
                    pad_len = abs(shift)
                    # Create silence padding matching the device of your audio tensors
                    padding = torch.zeros((*windows.shape[:-1], pad_len), dtype=windows.dtype, device=windows.device)
                    
                    if shift > 0:
                        # Shift right: Prepend silence, truncate the right edge
                        augmented_windows = torch.cat([padding, windows[..., :-pad_len]], dim=-1)
                    else:
                        # Shift left: Append silence, truncate the left edge
                        augmented_windows = torch.cat([windows[..., pad_len:], padding], dim=-1)

            elif aug == "add_noise" and noise_files:
                noise_path = random.choice(noise_files)
                # Load & match sample rate (logic from your pipeline)
                noise_audio, sr = torchaudio.load(noise_path)
                if sr != pipeline_ref.target_sr:
                    noise_audio = AF.resample(noise_audio, sr, pipeline_ref.target_sr)
                
                # Mixing using your SNR logic
                augmented_windows = pipeline_ref.add_noise_snr(
                    augmented_windows, 
                    noise_audio.mean(0, keepdim=True), 
                    snr_db=random.uniform(5, 20)
                )

            elif aug == "pitch_shift":
                # Note: PitchShift is slow on raw windows; usually done on Spectrograms
                n_steps = random.uniform(-2.0, 2.0)
                augmented_windows = AT.PitchShift(pipeline_ref.target_sr, n_steps)(augmented_windows)

        return augmented_windows




class PipistrelleDataset(Dataset):
    def __init__(
        self, 
        data_input : str | pd.DataFrame, 
        root_dir : str, 
        noise_folder : str | None =None, 
        is_training : bool =False,
        resample : bool = False, 
        resample_augment=None,
        online_augment = None,
        time_shift :bool = False,
        filter_echo : bool = False,
        overlap : float = 0.5,
        encoder : str ="perch2"
    ) -> None :
        
        self.root_dir = root_dir
        self.is_training = is_training

        #Wether resampling happens or not
        self.resample = resample
        #First element indicates if data augmentation during resampling should happen, other elements are the augmentations to apply during resampling
        self.resample_augment = resample_augment or [False]
        #First element indicates if online augmentation should happen, other elements are the augmentations to apply during online augmentation
        self.online_augment = online_augment or [False]
        self.time_shift = time_shift
        self.filter_echo = filter_echo

        self.augmentation_pipeline = AugmentationPipeline(online_augment, resample_augment)

         # Initialize processing pipeline
        if encoder == "perch2":
            self.pipeline = BatAudioPipeline(target_sr=32000, expansion_factor=5,window_sec =5,overlap=overlap, filter_echo=filter_echo)
        elif encoder == "effnetb0" or encoder == "NLM_BEATs":
            self.pipeline = BatAudioPipeline(target_sr=16000, expansion_factor=10,window_sec =10,overlap=overlap, filter_echo=filter_echo)
        else:
            raise ValueError("Unsupported encoder")

        # 1. Load raw data
        if isinstance(data_input, str):
            df = pd.read_csv(data_input)
        else:
            df = data_input # Assume it's a DataFrame

        # 2. Extract labels for mindful augmentation and balancing
        label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
        X_paths = df['relative_path'].values
        y_labels = df[label_cols].values

        # 3. Resample data
        self.original_len = len(X_paths)
        if self.is_training and self.resample: 
            self.X, self.y = self.augmentation_pipeline.iterative_oversample(X_paths, y_labels)
        else:
            self.X, self.y = X_paths, y_labels

        # Pre-load a list of noise files for augmentation
        self.noise_files = []
        if noise_folder and os.path.exists(noise_folder):
            self.noise_files = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.X[idx])
        labels = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # Grab a random noise file if we have them
        noise_path = random.choice(self.noise_files) if self.noise_files else None
        
        # Run the entire audio preprocessing and augmentation pipeline
        # 1. Grab the raw audio windows [Num_Windows, 160000]
        windows = self.pipeline.forward(file_path)

        if self.is_training:
            # If idx >= original_len, this is a cloned/resampled sample
            if idx >= self.original_len:
                # Apply "Resample Augmentations"
                windows = self.pipeline.forward(file_path,timeshift= self.time_shift)
                windows = self.augmentation_pipeline.augment_sample(
                    windows, self.resample_augment, self.noise_files, self.pipeline
                )
            
            # Apply standard "Online Augmentations" to everyone
            windows = self.augmentation_pipeline.augment_sample(
                windows, self.online_augment, self.noise_files, self.pipeline
            )

        return windows, labels
    
"""
# 1. Load the full CSV
df = pd.read_csv("full_dataset.csv")
label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']

# 2. Perform a Stratified Split (e.g., 80% train, 20% validation)
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the indices
for train_index, val_index in msss.split(df['relative_path'], df[label_cols]):
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[val_index].reset_index(drop=True)
"""