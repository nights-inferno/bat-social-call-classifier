import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
def plot_abmil_attention_on_pure_recordings(abmil_results, X_bags, y_true, target_indices, class_names, predict_abmil_fn):
    """
    Extracts the best model from Trial 0, Fold 0, runs inference on 4 pure 
    recordings using the pipeline's built-in scaling, and plots label-specific attention.
    """
    if len(target_indices) != 4:
        raise ValueError("Please provide exactly 4 indices for the target pure recordings.")

    # --- Step 1: Extract the Wrapper from Trial 0, Fold 0 ---
    best_wrapper = abmil_results[0]['best_models'][0]
    
    pt_model = best_wrapper.model_
    scaler = best_wrapper.scaler_
    device = best_wrapper.device
    
    if hasattr(pt_model, 'eval'):
        pt_model.eval()

    # --- Step 2: Set up Plot Canvas ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
    
    # --- Step 3: Run Inference and Extract Attention Weights ---
    for i, (idx, ax) in enumerate(zip(target_indices, axes)):
        bag_data = X_bags[idx]
        true_labels = y_true[idx]
        
        # Detect the target class index for this pure recording
        pure_class_idx = np.argmax(true_labels)
        target_class_name = class_names[pure_class_idx]
        
        # Run inference using your pipeline's native function
        # Note: we wrap bag_data in a list because predict_abmil loops over bags
        _, attention_out = predict_abmil_fn(pt_model, [bag_data], scaler, device=device)
        
        # --- FIX: Dictionary Unpacking Lineage ---
        # attention_out is structured as { 0: [label_0_arr, label_1_arr, ...] }
        if isinstance(attention_out, dict) and 0 in attention_out:
            bag_attention_list = attention_out[0]  # Extract list of arrays for bag 0
            
            # Select the attention timeline corresponding to our clean target class
            if isinstance(bag_attention_list, (list, tuple)) and len(bag_attention_list) > pure_class_idx:
                attention_profile = bag_attention_list[pure_class_idx]
            else:
                attention_profile = bag_attention_list
        else:
            attention_profile = attention_out
            
        # Ensure it's a completely flat, standard float64 numpy array for matplotlib
        attention_profile = np.asarray(attention_profile, dtype=np.float64).flatten()
        instances = np.arange(len(attention_profile), dtype=np.float64)
        
        # --- Step 4: Render Visual Timelines ---
        ax.fill_between(instances, attention_profile, color='#1f77b4', alpha=0.3)
        ax.plot(instances, attention_profile, color='#1f77b4', linewidth=1.5, label='Attention Weight')
        
        # Subplot Polish
        ax.set_title(f"Recording Index: {idx} | Verified Clean Target: {target_class_name}", 
                     fontsize=11, fontweight='bold', loc='left', pad=6)
        ax.set_ylabel("Attention", fontsize=9)
        ax.set_xlim(0, len(instances) - 1)
        
        # Prevent division by zero or empty limits if profile is flat (e.g. uniform hybrid mean pool)
        max_val = max(attention_profile) if len(attention_profile) > 0 else 1.0
        ax.set_ylim(0, (max_val * 1.15) if max_val > 0 else 1.0)  
        
        if i == 3:
            ax.set_xlabel("Instance Index (Time / Frame Sequence)", fontsize=10, fontweight='bold')

    plt.suptitle("ABMIL Instance-Level Attention Allocation Across Pure Acoustic Tracks", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa 

def plot_flashy_abmil_waveforms(abmil_results, X_bags, y_true, target_indices, class_names, 
                                predict_abmil_fn, root_dir, data_input_csv):
    """
    Loads raw audio files, upsamples frame-level attention weights, and plots
    high-contrast audio waveforms over a glowing, intense attention heatmap backdrop.
    """
    if len(target_indices) != 4:
        raise ValueError("Please provide exactly 4 indices for the target pure recordings.")

    # Load your tracking CSV file to map paths
    df_meta = pd.read_csv(data_input_csv)

    # Extract the trained model wrapper
    best_wrapper = abmil_results[0]['best_models'][0]
    pt_model = best_wrapper.model_
    scaler = best_wrapper.scaler_
    device = best_wrapper.device
    
    if hasattr(pt_model, 'eval'):
        pt_model.eval()

    # Create figure canvas
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
    
    for i, (idx, ax) in enumerate(zip(target_indices, axes)):
        # --- Step 1: Resolve File Path and Load Raw Audio ---
        rel_path = df_meta['relative_path'].values[idx]
        file_path = os.path.join(root_dir, rel_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not locate audio file at: {file_path}")
        y_audio, sr = librosa.load(file_path, sr=None)
        
        duration = len(y_audio) / sr
        time_axis = np.linspace(0, duration, len(y_audio))

        # --- Step 2: Get Model Attention Profile ---
        bag_data = X_bags[idx]
        true_labels = y_true[idx]
        pure_class_idx = np.argmax(true_labels)
        target_class_name = class_names[pure_class_idx]
        
        _, attention_out = predict_abmil_fn(pt_model, [bag_data], scaler, device=device)
        
        # Unpack the nested dictionary/list structure from your predict_abmil function
        if isinstance(attention_out, dict) and 0 in attention_out:
            bag_attention_list = attention_out[0]
            if isinstance(bag_attention_list, (list, tuple)) and len(bag_attention_list) > pure_class_idx:
                attention_profile = bag_attention_list[pure_class_idx]
            else:
                attention_profile = bag_attention_list
        else:
            attention_profile = attention_out
            
        attention_profile = np.asarray(attention_profile, dtype=np.float64).flatten()

        # --- Step 3: Smoothly Upsample Attention to Match Audio Timeline ---
        attention_upsampled = np.interp(
            time_axis, 
            np.linspace(0, duration, len(attention_profile)), 
            attention_profile
        )
        
        # Normalize attention profile to [0, 1] for maximum colormap dynamic range split
        att_min, att_max = attention_upsampled.min(), attention_upsampled.max()
        if (att_max - att_min) > 1e-8:
            attention_norm = (attention_upsampled - att_min) / (att_max - att_min)
        else:
            attention_norm = attention_upsampled

        # --- Step 4: UI styling & Intense Render Pipeline ---
        ax.set_facecolor('#0d0e15') 
        ax.grid(False) 
        
        # Draw the glowing background backdrop matrix using the 'inferno' flame colormap
        y_bounds = [y_audio.min() * 1.2, y_audio.max() * 1.2]
        ax.imshow(
            attention_norm.reshape(1, -1), 
            cmap='inferno', 
            aspect='auto',
            extent=[0, duration, y_bounds[0], y_bounds[1]],
            alpha=0.75, 
            zorder=1
        )
        
        # Overlay the high-contrast crisp white audio waveform directly over the glowing energy trail
        ax.plot(time_axis, y_audio, color='#FFFFFF', linewidth=0.7, alpha=0.9, zorder=2)

        # Polish labels and titles
        ax.set_title(f"Track [{idx}] Waveform  |  Verified Target: {target_class_name.upper()}", 
                     color='#e0e0e6', fontsize=11, fontweight='bold', loc='left', pad=6)
        ax.set_ylabel("Amplitude", color='#a0a0ab', fontsize=9)
        ax.tick_params(colors='#a0a0ab', labelsize=9)
        ax.set_xlim(0, duration)
        ax.set_ylim(y_bounds[0], y_bounds[1])

        if i == 3:
            ax.set_xlabel("Time (Seconds)", color='#e0e0e6', fontsize=10, fontweight='bold')

    # Overall figure canvas styling adjustments
    fig.patch.set_facecolor('#0d0e15')
    
    # --- FIXED LINE: Removed letterspacing parameter ---
    plt.suptitle("MIL ATTENTION-MAPPED AUDIO WAVEFORMS (NEON GLOW INDICATES HIGHEST MODEL ATTENTION)", 
                 color='#ffffff', fontsize=13, fontweight='bold', y=0.98)
                 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa 

def plot_flashy_abmil_spectrograms(abmil_results, X_bags, y_true, target_indices, class_names, 
                                   predict_abmil_fn, root_dir, data_input_csv):
    """
    Loads raw audio files, computes Mel Spectrograms, and dynamically illuminates
    the spectrogram frequencies based on the model's attention weights.
    """
    if len(target_indices) != 4:
        raise ValueError("Please provide exactly 4 indices for the target pure recordings.")

    # Load tracking metadata CSV
    df_meta = pd.read_csv(data_input_csv)

    # Extract model variables
    best_wrapper = abmil_results[0]['best_models'][0]
    pt_model = best_wrapper.model_
    scaler = best_wrapper.scaler_
    device = best_wrapper.device
    
    if hasattr(pt_model, 'eval'):
        pt_model.eval()

    # Set up the dark studio canvas
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
    
    for i, (idx, ax) in enumerate(zip(target_indices, axes)):
        # --- Step 1: Resolve Paths & Load Audio Track ---
        rel_path = df_meta['relative_path'].values[idx]
        file_path = os.path.join(root_dir, rel_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not locate audio file at: {file_path}")
        y_audio, sr = librosa.load(file_path, sr=None)
        
        duration = len(y_audio) / sr

        # --- Step 2: Compute Mel Spectrogram ---
        S = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        n_mels, n_frames = S_db.shape
        
        # Normalize spectrogram matrix to [0, 1] for colormap assignment
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        
        # Map the normalized values into an RGBA image matrix
        spec_rgba = plt.cm.inferno(S_norm)

        # --- Step 3: Extract and Align Attention Profile ---
        bag_data = X_bags[idx]
        true_labels = y_true[idx]
        pure_class_idx = np.argmax(true_labels)
        target_class_name = class_names[pure_class_idx]
        
        _, attention_out = predict_abmil_fn(pt_model, [bag_data], scaler, device=device)
        
        # Safe dictionary/list unpacking from your custom ABMIL tracking system
        if isinstance(attention_out, dict) and 0 in attention_out:
            bag_attention_list = attention_out[0]
            if isinstance(bag_attention_list, (list, tuple)) and len(bag_attention_list) > pure_class_idx:
                attention_profile = bag_attention_list[pure_class_idx]
            else:
                attention_profile = bag_attention_list
        else:
            attention_profile = attention_out
            
        attention_profile = np.asarray(attention_profile, dtype=np.float64).flatten()

        # Interpolate low-res attention tokens to align perfectly with the spectrogram's time frames
        time_axis_spec = np.linspace(0, duration, n_frames)
        attention_upsampled = np.interp(
            time_axis_spec, 
            np.linspace(0, duration, len(attention_profile)), 
            attention_profile
        )
        
        # Normalize the attention array for intensity mapping
        att_min, att_max = attention_upsampled.min(), attention_upsampled.max()
        if (att_max - att_min) > 1e-8:
            att_norm = (attention_upsampled - att_min) / (att_max - att_min)
        else:
            att_norm = np.ones_like(attention_upsampled)

        # --- Step 4: Perform the Intensity Flash Processing ---
        # Baseline brightness is 15% (shadowy backdrop), scaling up to 100% burst illumination
        intensity_profile = 0.15 + 0.85 * att_norm
        
        # Multiply RGB values across all Mel rows by the time-varying intensity vector
        spec_rgba[:, :, :3] *= intensity_profile[np.newaxis, :, np.newaxis]

        # --- Step 5: Render Visual Elements ---
        ax.set_facecolor('#0d0e15')
        ax.grid(False)
        
        # Render the illuminated spectrogram
        ax.imshow(
            spec_rgba, 
            origin='lower', 
            aspect='auto', 
            extent=[0, duration, 0, n_mels],
            zorder=1
        )
        
        # Polish subplot elements
        ax.set_title(f"Track [{idx}] Spectrogram  |  Verified Target: {target_class_name.upper()}", 
                     color='#e0e0e6', fontsize=11, fontweight='bold', loc='left', pad=6)
        ax.set_ylabel("Mel Frequency Bands", color='#a0a0ab', fontsize=9)
        ax.tick_params(colors='#a0a0ab', labelsize=9)
        ax.set_xlim(0, duration)

        if i == 3:
            ax.set_xlabel("Time (Seconds)", color='#e0e0e6', fontsize=10, fontweight='bold')

    # Global canvas styling
    fig.patch.set_facecolor('#0d0e15')
    plt.suptitle("ABMIL SPECTROGRAM FLASH-ILLUMINATION (BURSTS HIGHLIGHT CLASSIFICATION CRITICAL BIOACOUSTIC TOKENS)", 
                 color='#ffffff', fontsize=12, fontweight='bold', y=0.98)
                 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()