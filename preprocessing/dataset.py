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

    def load_and_expand(self, file_path : str) -> torch.Tensor:
        """Loads audio using soundfile and applies 10x time expansion logic."""
        try:
            # Load using soundfile
            data, orig_sr = sf.read(file_path)

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

            # Time Expansion logic
            virtual_sr = orig_sr // self.expansion_factor

            # Resample to target sample rate
            if virtual_sr != self.target_sr:
                audio = AF.resample(audio, orig_freq=virtual_sr, new_freq=self.target_sr)

            return audio

        except Exception as e:
            print(f"\n[Warning] Skipping {file_path}: {e}")
            # Return 1 second of silence as a fallback to prevent NoneType errors
            return torch.zeros((1, self.target_sr))

    def apply_bandpass(self, audio : torch.Tensor) -> torch.Tensor:
        """
        Filters frequencies bats don't emit, taking into account time expansion.
        The high-cut is already handled by the resampling Nyquist limit.
        Eventually cuts out echolocation frequency band
        """
        if self.filter_echo:
            fmin = 40000/self.expansion_factor
            fmax = 75000/self.expansion_factor
            fcenter = np.sqrt(fmin * fmax)
            Q = fcenter / (fmax - fmin)
            audio = AF.bandreject_biquad(audio, sample_rate=self.target_sr, central_freq=fcenter, Q=Q)
        # Highpass biquad filter at 15/time_expansion_factor kHz (Expanded domain)
        return AF.highpass_biquad(audio, sample_rate=self.target_sr, cutoff_freq=15000/self.expansion_factor)

    def generate_colored_noise(self, num_samples : int, exponent=1.0) -> torch.Tensor:
        """0.0=White, 1.0=Pink (Rain), 2.0=Brown (Roar)"""
        white_noise_fft = torch.fft.rfft(torch.randn(num_samples))
        frequencies = torch.fft.rfftfreq(num_samples)
        # Apply power law 1/f^beta
        scaler = 1.0 / (frequencies** (exponent / 2.0) + 1e-10)
        noise = torch.fft.irfft(white_noise_fft * scaler, n=num_samples)
        return (noise / (noise.std() + 1e-10)).unsqueeze(0)
    
    def add_noise_snr(self, audio : torch.Tensor, noise_audio : torch.Tensor, snr_db : float) -> torch.Tensor:
        """Mixes background noise at a specific Signal-to-Noise Ratio."""
        # Ensure noise is the same length as the audio
        if noise_audio.shape[1] < audio.shape[1]:
            # Repeat noise if it's too short
            repeats = math.ceil(audio.shape[1] / noise_audio.shape[1])
            noise_audio = noise_audio.repeat(1, repeats)
        
        # Trim noise to exact audio length
        noise_audio = noise_audio[:, :audio.shape[1]]
        
        # Calculate powers
        audio_power = audio.norm(p=2)
        noise_power = noise_audio.norm(p=2)
        
        # Avoid division by zero
        if noise_power == 0:
            return audio
            
        # Calculate required noise scalar to match target SNR
        # SNR = 20 * log10(audio_power / target_noise_power)
        target_noise_power = audio_power / (10 ** (snr_db / 20.0))
        noise_scalar = target_noise_power / noise_power
        
        # Mix
        mixed_audio = audio + (noise_audio * noise_scalar)
        return mixed_audio


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

    def Z_normalize_windows(self, windows : torch.Tensor) -> torch.Tensor:
        mean = windows.mean(dim=-1, keepdim=True)
        std = windows.std(dim=-1, keepdim=True)
        normalized_windows = (windows - mean) / (std + 1e-8)
        return normalized_windows

    def forward(self, file_path : str) -> torch.Tensor:

        # 0. Load & Time Expand
        audio = self.load_and_expand(file_path)

        # 1. Noise reduce
        # (Implementation for noise reduction would go here)
        
        # 2. Bandpass Filter
        audio = self.apply_bandpass(audio)

        # 3. Cut into Windows
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
        
    def iterative_oversample(self, X, y, target_percentage=1.0):
        """
        Randomly duplicates samples containing minority labels 
        until the distribution balances out.
        """
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

            idx_to_clone = np.random.choice(possible_indices)

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
                shift = random.randint(-int(0.1 * windows.shape[-1]), int(0.1 * windows.shape[-1]))
                augmented_windows = torch.roll(augmented_windows, shifts=shift, dims=-1)

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
        # Returns shape: [Num_Windows, 1, 128, Frames]
        windows_tensor = self.pipeline.forward(file_path)

        if self.is_training:
            # If idx >= original_len, this is a cloned/resampled sample
            if idx >= self.original_len:
                # Apply "Resample Augmentations"
                windows = self.augmentation_pipeline.augment_sample(
                    windows_tensor, self.resample_augment, self.noise_files, self.pipeline
                )
            
            # Apply standard "Online Augmentations" to everyone
            windows = self.augmentation_pipeline.augment_sample(
                windows_tensor, self.online_augment, self.noise_files, self.pipeline
            )

        return windows_tensor, labels
    
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