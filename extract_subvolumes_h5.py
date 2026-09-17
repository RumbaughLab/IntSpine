import os
import h5py
import numpy as np
import scipy.signal
import tifffile

def extract_subvolumes_from_h5(
    h5_path, 
    output_dir=None, 
    only_selected=True,
    percentile_val=99.0, 
    min_prominence_ratio=0.05, 
    peak_rel_height=0.85, 
    z_buffer=2,
    xpixdim=0.108, 
    ypixdim=0.108, 
    zpixdim=0.3
):
    """
    Extracts each segment volume from an IntSegment HDF5 file, applies
    Z-axis peak prominence cropping, and exports them as calibrated 3D TIFFs.
    """
    if output_dir is None:
        base_dir = os.path.dirname(h5_path)
        output_dir = os.path.join(base_dir, "cropped_subvolumes")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(h5_path).replace("_extracted.h5", "")

    with h5py.File(h5_path, "r") as h5f:
        # Collect and sort all segment keys (segment_1, segment_2, ...)
        segment_keys = [k for k in h5f.keys() if k.startswith("segment_")]
        segment_keys.sort(key=lambda k: int(k.split("_")[1]))

        print(f"Found {len(segment_keys)} segments in {os.path.basename(h5_path)}")

        for key in segment_keys:
            grp = h5f[key]
            seg_id = int(key.split("_")[1])
            is_selected = grp.attrs.get("is_selected", False)

            # Skip unselected segments if only_selected is True
            if only_selected and not is_selected:
                continue

            # 1. Load the XY-cropped 3D subvolume stored in HDF5 (shape: Z, Y, X)
            xy_cropped_stack = grp["volume"][:]
            max_z = xy_cropped_stack.shape[0]

            # 2. Z-Cropping Logic via Peak Prominence
            pixel_threshold = np.percentile(xy_cropped_stack, float(percentile_val))
            masked_volume = np.where(xy_cropped_stack >= pixel_threshold, xy_cropped_stack, 0)
            z_sums = np.sum(masked_volume, axis=(1, 2))

            min_prominence = np.max(z_sums) * min_prominence_ratio
            peaks, properties = scipy.signal.find_peaks(z_sums, prominence=min_prominence)

            if len(peaks) == 0:
                print(f"Segment {seg_id}: No signal peaks found. Saving full Z-range.")
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

            # 3. Final Z-Cropped 3D Array
            final_subvolume = xy_cropped_stack[start_z:end_z, :, :]

            # 4. Save to ImageJ-compatible 3D TIFF with spatial calibration
            cluster_name = grp.attrs.get("cluster_name", "Unknown")
            out_filename = f"{base_name}_Seg{seg_id}_{cluster_name}_Z{start_z}-{end_z-1}.tif"
            out_path = os.path.join(output_dir, out_filename)

            tifffile.imwrite(
                out_path,
                final_subvolume,
                imagej=True,
                resolution=(1.0 / xpixdim, 1.0 / ypixdim),
                metadata={'spacing': zpixdim, 'unit': 'um'}
            )

            print(f"Saved: {out_filename} (Shape: {final_subvolume.shape})")