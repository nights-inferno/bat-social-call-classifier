
import os
import re
import pandas as pd
from pathlib import Path

def consolidate_ood_dataset(root_dir, output_csv="ood_metadata.csv"):
    """
    Build metadata CSV for OOD domain transfer dataset.
    
    Folder structure:
        root/
            echolocation/
            type A/
            type B/
                in flight/
                roosting/
            type C/
                C.i/
                C.d stationnary/
                C.d flying/
                C.d advertisement/
                Noctule/
                General/
            type D/
                Noctule/
                Thrills Nathusius/
                Part D Nathusius/
                Part E Nathusius/
    
    Filename convention: XC826467 - Nathusius's Pipistrelle - Pipistrellus nathusii.wav
    """
    
    class_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    
    # ── Top-level folder → coarse label mapping ──────────────
    # Specialized subfolders all map to their parent label
    top_level_map = {
        'type A':        [1, 0, 0, 0, 0],
        'type B':        [0, 1, 0, 0, 0],
        'type C':        [0, 0, 1, 0, 0],
        'type D':        [0, 0, 0, 1, 0],
        'echolocation':  [0, 0, 0, 0, 1],
    }
    
    # ── Subtype label for fine-grained analysis ───────────────
    # Maps (top_level_folder, subfolder) → subtype string
    # Top-level files (no subfolder) get subtype = 'general'
    subtype_map = {
        ('type B', 'in flight'):              'B.in_flight',
        ('type B', 'roosting'):               'B.roosting',
        ('type C', 'C.i'):                    'C.i',
        ('type C', 'C.d stationnary'):        'C.d_stationary',
        ('type C', 'C.d flying'):             'C.d_flying',
        ('type C', 'C.d advertisement'):      'C.d_advertisement',
        ('type C', 'Noctule'):                'C.noctule',
        ('type C', 'General'):                'C.general',
        ('type D', 'Noctule'):                'D.noctule',
        ('type D', 'Thrills Nathusius'):      'D.thrills_nathusius',
        ('type D', 'Part D Nathusius'):       'D.part_d_nathusius',
        ('type D', 'Part E Nathusius'):       'D.part_e_nathusius',
    }
    
    all_records = []
    
    for top_folder, label_vector in top_level_map.items():
        top_path = os.path.join(root_dir, top_folder)
        if not os.path.exists(top_path):
            print(f"Skipping missing folder: {top_folder}")
            continue
        
        # Walk the full subtree
        for dirpath, dirnames, filenames in os.walk(top_path):
            # Determine subfolder name relative to top_path
            rel = os.path.relpath(dirpath, top_path)
            subfolder = None if rel == '.' else rel
            
            # Get subtype label
            if subfolder is not None:
                subtype = subtype_map.get(
                    (top_folder, subfolder), 
                    f'{top_folder[:1].lower()}.{subfolder.lower()}'  # fallback
                )
            else:
                subtype = top_folder.replace(' ', '_').lower()
            
            for fname in filenames:
                if not fname.lower().endswith(('.wav', '.mp3', '.flac')):
                    continue
                
                # Parse filename: XC826467 - Common Name - Genus species.wav
                species_common, species_latin = parse_filename(fname)
                
                record = {
                    'filename':       fname,
                    'top_folder':     top_folder,
                    'subfolder':      subfolder,
                    'subtype':        subtype,
                    'species_common': species_common,
                    'species_latin':  species_latin,
                    'relative_path':  os.path.join(
                        os.path.relpath(dirpath, root_dir), fname
                    ),
                }
                for i, col in enumerate(class_cols):
                    record[col] = label_vector[i]
                
                all_records.append(record)
    
    if not all_records:
        print("No audio files found.")
        return
    
    df = pd.DataFrame(all_records)
    
    # ── Merge multi-label duplicates (same filename in multiple folders) ──
    agg_dict = {col: 'max' for col in class_cols}
    agg_dict.update({
        'top_folder':     lambda x: '+'.join(sorted(x.unique())),
        'subfolder':      'first',
        'subtype':        lambda x: '+'.join(sorted(x.unique())),
        'species_common': 'first',
        'species_latin':  'first',
        'relative_path':  'first',
    })
    
    df_merged = (
        df.groupby('filename')
          .agg(agg_dict)
          .reset_index()
          .sort_values('filename')
    )
    
    # ── Summary ───────────────────────────────────────────────
    print("-" * 40)
    print(f"Unique files:        {len(df_merged)}")
    print(f"Multi-label files:   {(df_merged[class_cols].sum(axis=1) > 1).sum()}")
    print(f"\nLabel distribution:")
    print(df_merged[class_cols].sum())
    print(f"\nSpecies found:")
    print(df_merged['species_latin'].value_counts().to_string())
    print(f"\nSubtypes found:")
    print(df_merged['subtype'].value_counts().to_string())
    
    df_merged.to_csv(output_csv, index=False)
    print(f"\nCSV saved: {output_csv}")
    
    return df_merged


def parse_filename(fname):
    """
    Parse: 'XC826467 - Nathusius's Pipistrelle - Pipistrellus nathusii.wav'
    Returns: (common_name, latin_name)
    """
    # Remove extension
    stem = os.path.splitext(fname)[0]
    
    # Split on ' - '
    parts = [p.strip() for p in stem.split(' - ')]
    
    if len(parts) >= 3:
        # parts[0] = XC ID, parts[1] = common name, parts[2] = latin name
        return parts[1], parts[2]
    elif len(parts) == 2:
        return parts[1], None
    else:
        return None, None


# Run
dir = Path(os.getcwd()).resolve().parent / "data"
df = consolidate_ood_dataset(
    r'C:\Users\artem\Nast. Code\new-git\bat-social-call-classifier\data\domain-transfer-dataset',
    output_csv="ood_metadata.csv"
)