import os
import re
import shutil
from pathlib import Path

def batch_copy_mip_files(
    source_root_dir: str,
    destination_dir: str = "MIP",
    file_pattern: str = "*projection*.tif"
):
    """
    Recursively finds all projection/MIP files across the dataset tree,
    extracts metadata (Slide_well, neuron, region, timepoint),
    and copies them to the destination directory with standardized names.
    
    Example input path:
      ...\\Slide12_well2\\n1\\basal\\experiment\\timepoint0\\...projection_t0...tif
    
    Output file:
      MIP/Slide12_well2-n1-basal-t0.tif
    """
    src_root = Path(source_root_dir)
    dest_folder = Path(destination_dir)
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Searching for files matching '{file_pattern}' in:\n{src_root}")
    print(f"Destination directory: {dest_folder.resolve()}")
    print(f"=" * 60)
    
    if not src_root.exists():
        print(f"\n[Error] Source directory not found: {src_root}")
        print("Please check drive mapping (e.g., Z:) and network connection.")
        return
    
    matching_files = list(src_root.rglob(file_pattern))
    print(f"Found {len(matching_files)} matching file(s).\n")
    
    copied_count = 0
    skipped_count = 0
    
    for file_path in matching_files:
        try:
            parts = file_path.parts
            
            # Find indices for relevant folder components in path:
            # Hierarchy: [..., Slide_well, neuron, region, experiment, timepointX, filename]
            # Search for 'timepoint' folder in parts
            tp_match = None
            tp_idx = -1
            for idx, part in enumerate(parts):
                if "timepoint" in part.lower() or re.match(r"^t\d+$", part.lower()):
                    tp_match = part
                    tp_idx = idx
                    break
            
            if tp_idx != -1 and tp_idx >= 4:
                # Structure relative to timepoint:
                # parts[tp_idx - 4] = Slide_well (e.g. Slide12_well2)
                # parts[tp_idx - 3] = neuron     (e.g. n1)
                # parts[tp_idx - 2] = region     (e.g. basal)
                # parts[tp_idx]     = timepoint  (e.g. timepoint0)
                slide_well = parts[tp_idx - 4]
                neuron = parts[tp_idx - 3]
                region = parts[tp_idx - 2]
                timepoint = tp_match.lower().replace("timepoint", "t")
            else:
                # Fallback: relative to file position (parts[-6], parts[-5], parts[-4], parts[-2])
                slide_well = parts[-6]
                neuron = parts[-5]
                region = parts[-4]
                timepoint = parts[-2].lower().replace("timepoint", "t")
            
            # Construct new filename
            new_filename = f"{slide_well}-{neuron}-{region}-{timepoint}{file_path.suffix}"
            dest_file = dest_folder / new_filename
            
            # Copy while preserving original files & timestamps
            shutil.copy2(file_path, dest_file)
            print(f"[COPIED] {file_path.name}")
            print(f"     -> {dest_file.name}")
            copied_count += 1
            
        except Exception as e:
            print(f"[SKIPPED] Could not process '{file_path}': {e}")
            skipped_count += 1
            
    print(f"\n" + "=" * 60)
    print(f"Batch Processing Complete!")
    print(f"Total Copied : {copied_count}")
    print(f"Total Skipped: {skipped_count}")
    print(f"Output folder: {dest_folder.resolve()}")
    print(f"=" * 60)

if __name__ == "__main__":
    # Dataset root directory containing all slides/wells
    dataset_root = r"Z:\2026 - E19 Project collab spine and dendrites\Spine\60x_dataset"
    
    # Destination folder name or path
    destination_folder = r"MIP"
    
    # File pattern to search (e.g., *projection*.tif or *.tif)
    pattern = "*projection*.tif"
    
    batch_copy_mip_files(
        source_root_dir=dataset_root,
        destination_dir=destination_folder,
        file_pattern=pattern
    )

