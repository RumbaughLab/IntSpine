import os
import re
import h5py
import numpy as np
import scipy.signal
import tifffile
from pathlib import Path

def natural_sort_key(s):
    """Sorts strings with embedded numbers naturally (e.g., z2 before z10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def load_z_stack_from_dir(directory):
    """Loads a directory of 2D TIFF slices into a 3D NumPy array."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Z-stack directory not found: {directory}")
        
    files = list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {directory}")
        
    # Natural sort to ensure z1, z2, ..., z10 order is correct
    files.sort(key=lambda f: natural_sort_key(f.name))
    
    slices = []
    for f in files:
        img = tifffile.imread(f)
        slices.append(img)
        
    return np.stack(slices)

def parse_original_z_stack_path(base_name, dataset_root):
    """
    Parses a standardized MIP name (e.g., Slide9_well1-n1-basal-t0)
    to its original experiment_z_stack path.
    """
    parts = base_name.split('-')
    if len(parts) != 4:
        raise ValueError(f"Filename does not match expected pattern (Slide_well-neuron-region-timepoint): {base_name}")
        
    slide_well, neuron, region, tp = parts
    # Convert 't0' to 'timepoint0'
    timepoint_str = tp.replace('t', 'timepoint')
    
    # Reconstruct the Z-stack directory path
    z_dir = Path(dataset_root) / slide_well / neuron / region / "experiment_z_stack" / timepoint_str
    return z_dir

def process_segments_for_file(
    h5_path, 
    dataset_root, 
    output_dir,
    percentile_val=99.0, 
    min_prominence_ratio=0.05, 
    peak_rel_height=0.85, 
    z_buffer=2,
    xpixdim=0.108, 
    ypixdim=0.108, 
    zpixdim=0.3
):
    """Reads HDF5 coordinates, loads raw Z-stack, crops in 3D, and exports TIFFs."""
    h5_path = Path(h5_path)
    base_name = h5_path.name.replace("_extracted.h5", "")
    
    try:
        z_dir = parse_original_z_stack_path(base_name, dataset_root)
    except Exception as e:
        print(f"[Error] {e}")
        return
        
    print(f"\nProcessing: {base_name}")
    print(f"Loading Z-Stack from: {z_dir}")
    
    try:
        full_z_stack = load_z_stack_from_dir(z_dir)
        print(f"Loaded Z-stack shape: {full_z_stack.shape}")
    except Exception as e:
        print(f"[Error] Failed to load Z-stack: {e}")
        return

    with h5py.File(h5_path, "r") as h5f:
        segment_keys = [k for k in h5f.keys() if k.startswith("segment_")]
        segment_keys.sort(key=lambda k: int(k.split("_")[1]))
        
        extracted_count = 0
        for key in segment_keys:
            grp = h5f[key]
            seg_id = int(key.split("_")[1])
            is_selected = grp.attrs.get("is_selected", False)
            
            if not is_selected:
                continue
                
            # 1. Get XY bounding box from IntSegment metadata
            # bbox is stored as [top, bottom, left, right]
            t, b, l, r = grp.attrs['bbox']
            
            # Clamp to image dimensions
            max_z, max_y, max_x = full_z_stack.shape
            t = max(0, min(max_y, t))
            b = max(0, min(max_y, b))
            l = max(0, min(max_x, l))
            r = max(0, min(max_x, r))

            # 2. Extract XY region from full Z-stack
            xy_cropped_stack = full_z_stack[:, t:b, l:r]
            
            # 3. Z-Cropping Logic via Peak Prominence
            pixel_threshold = np.percentile(xy_cropped_stack, float(percentile_val))
            masked_volume = np.where(xy_cropped_stack >= pixel_threshold, xy_cropped_stack, 0)
            z_sums = np.sum(masked_volume, axis=(1, 2))

            min_prominence = np.max(z_sums) * min_prominence_ratio
            peaks, properties = scipy.signal.find_peaks(z_sums, prominence=min_prominence)

            if len(peaks) == 0:
                print(f"  Segment {seg_id}: No signal peaks found. Saving full Z-range.")
                start_z, end_z = 0, max_z
            else:
                best_peak_idx = np.argmax(properties["prominences"])
                best_peak = peaks[best_peak_idx]

                widths, _, left_ips, right_ips = scipy.signal.peak_widths(
                    z_sums, [best_peak], rel_height=peak_rel_height
                )
                detected_start = int(left_ips[0])
                detected_end = int(right_ips[0])

                # Apply buffer and clamp to volume boundaries
                start_z = max(0, detected_start - z_buffer)
                end_z = min(max_z, detected_end + z_buffer + 1)

            # 4. Final Z-Cropped 3D Array
            final_subvolume = xy_cropped_stack[start_z:end_z, :, :]
            
            # 5. Save to TIFF
            # Formatting as filename-sXXX.tif
            out_filename = f"{base_name}-s{seg_id:03d}.tif"
            out_path = Path(output_dir) / out_filename
            
            tifffile.imwrite(
                out_path,
                final_subvolume,
                imagej=True,
                resolution=(1.0 / xpixdim, 1.0 / ypixdim),
                metadata={'spacing': zpixdim, 'unit': 'um'}
            )
            print(f"  -> Saved {out_filename} | Z-range: {start_z}-{end_z-1} | Shape: {final_subvolume.shape}")
            extracted_count += 1
            
        print(f"Completed {base_name}: Extracted {extracted_count} segments.")

def main():
    # Configure your paths here
    DATASET_ROOT = r"Z:\2026 - E19 Project collab spine and dendrites\Spine\60x_dataset"
    MIP_DIR = r"Z:\2026 - E19 Project collab spine and dendrites\Spine\60x_dataset\MIP"
    H5_DIR = os.path.join(MIP_DIR, "output_analysis")
    OUTPUT_SEGMENTED_DIR = r"MIP_segmented"
    
    Path(OUTPUT_SEGMENTED_DIR).mkdir(parents=True, exist_ok=True)
    
    print("============================================================")
    print("Starting Segment Extraction...")
    print(f"HDF5 Directory: {Path(H5_DIR).resolve()}")
    print(f"Z-Stack Root: {DATASET_ROOT}")
    print(f"Output Directory: {Path(OUTPUT_SEGMENTED_DIR).resolve()}")
    print("============================================================")
    
    if not Path(H5_DIR).exists():
        print(f"[Error] HDF5 directory does not exist: {H5_DIR}")
        return
        
    h5_files = list(Path(H5_DIR).glob("*_extracted.h5"))
    if not h5_files:
        print("No *_extracted.h5 files found to process.")
        return
        
    for h5_path in h5_files:
        process_segments_for_file(
            h5_path=h5_path,
            dataset_root=DATASET_ROOT,
            output_dir=OUTPUT_SEGMENTED_DIR
        )
        
    print("\nExtraction Complete!")

if __name__ == "__main__":
    main()

