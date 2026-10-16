import os
import shutil
from pathlib import Path

def copy_and_rename_file(src_path: str, dest_dir: str, new_name: str):
    """
    Copies a source file to the destination directory under a new name,
    preserving the original file and metadata.
    """
    src = Path(src_path)
    dest_folder = Path(dest_dir)
    
    # Ensure the destination directory exists (e.g. MIP)
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Retain the original file extension if not explicitly included in new_name
    ext = src.suffix
    if not new_name.endswith(ext):
        target_filename = f"{new_name}{ext}"
    else:
        target_filename = new_name
        
    dest_file = dest_folder / target_filename
    
    # Perform copy (shutil.copy2 preserves original file & timestamps)
    print(f"Copying:\n  Source: {src}\n  Destination: {dest_file}")
    
    if not src.exists():
        print(f"\n[Warning] Source file does not exist at path:\n{src}")
        print("Please check drive mapping (e.g., Z:) or file availability.")
        return dest_file

    shutil.copy2(src, dest_file)
    print("\n[Success] File copied successfully!")
    return dest_file

if __name__ == "__main__":
    # Specific file path requested
    source_file = r"Z:\2026 - E19 Project collab spine and dendrites\Spine\60x_dataset\Slide12_well2\n1\basal\experiment\timepoint0\Rumbaugh_e19_60X_targetoverview_projection_t0_B01_s0_w0_z0.tif"
    
    # Destination directory (MIP folder)
    destination_directory = r"MIP"
    
    # Desired new filename
    new_filename = "Slide12_well2-n1-basal-t0"
    
    copy_and_rename_file(source_file, destination_directory, new_filename)

