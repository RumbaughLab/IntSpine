
<img src="intspine_logo.png" width="60" title="IntSpine" alt="IntSpine" align="left" vspace = "50">

#   IntSpine: Interactive Spine Analysis Tool
 

## Overview
This repository contains a standalone Python desktop application (PySide6) and associated tools for volumetric spine quantification from z-stack images. It was developed to benchmark automated solutions against ground truth data, as existing tools were not optimal for high-resolution datasets. 

The application enables the easy extraction of volumes from z-stacks, facilitating manual annotation, eliminating z-axis bleeding, and drastically improving data traceability for laboratory colleagues.

![IntSpine User Interface](assets/workflow.png)
*(Caption: The IntSpine interface featuring 2D curvature isolation, real-time barrier painting, and interactive z-slice navigation.)*

## Features & Rationale
To facilitate analysis, we implemented a processing pipeline designed to:
*   **Improve data traceability** for rigorous quantitative microscopy.
*   **Enable automated processing and batching** of segments to be used with tools like RESPAN.
*   **Facilitate manual annotation** by rapidly loading isolated segments sequentially.
*   **Isolate morphology** using 2.5D extruded envelopes and 2D Hessian curvature filtering to completely prevent longitudinal shaft bleeding.

## Installation

This application requires a dedicated Conda environment to safely manage its scientific imaging dependencies.

**1. Create the Environment File**
Save the following configuration as `environment.yml` in your project directory:
```yaml
name: spine_analyzer
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - matplotlib
  - scipy
  - scikit-image
  - tifffile
  - pip
  - pip:
    - PySide6
```

**2. Build and Activate the Environment**
Open your terminal (or Anaconda Prompt) and execute the following commands:
```bash
# Navigate to the repository
cd /path/to/spine-manual-tool

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate spine_analyzer
```

**3. Launch the Application**
```bash
python spine_analyzer_app.py
```

## Workflow

![IntSpine User Interface](assets/UI1_screenshot.png)
*(Caption: Graphical representation of the pre-processing, tracing, and 2.5D volumetric segmentation pipeline.)*

### 1. Pre-Processing & ROI Extraction
This step speeds up the workflow by isolating segments before tracing.
1. Open a **z-max projection** of your z-stack. This is significantly faster to load than a full z-stack and makes it much easier to identify branches of interest.
2. Draw Regions of Interest (ROIs) on the z-projection.
3. Run the automated extraction tool. *(Note: Loading the full image—e.g., 500 z-steps—takes about 1 minute, but this process runs entirely automatically).*
4. The tool will automatically extract and crop all ROIs in X, Y, and Z dimensions.

### 2. Tracing and Masking
Once the ROIs are cropped and extracted, loading them for labeling is extremely fast.
1. Load the extracted ROI segments.
2. Trace the neurite using **SNT (Simple Neurite Tracer)**.
3. **Important Calibration**: Ensure the image is properly calibrated to the correct pixel/µm dimensions before tracing. 
4. **Diameter Masking**: While tracing the neurite, use the scroll wheel to adjust and capture the *actual diameter* of the dendrite. This trace acts as a spatial barrier during z-quantification.
5. Save the completed trace as an `.swc` file (or generate a Geo/Respan mask).

### 3. Volumetric Quantification (IntSpine App)
1. **Load Data:** Launch `spine_analyzer_app.py`. Click **Browse Folder** to select the directory containing your cropped `.tif` images and their corresponding masks/traces. Click **Load Remaining**.
2. **Select Mask Source:** Choose the appropriate barrier mask (`SWC Mask`, `Geo Mask`, or `Respan Mask`) from the dropdown. Adjust the **Barrier µm** slider to dilate or contract the dendritic exclusion zone.
3. **Generate Seed Targets:** Populate the target queue using one of three methods:
    *   **Manual Seeding:** Left-click on the Z-Slice Navigator or Global MIP. The algorithm automatically snaps to the optimal local Z-plane.
    *   **Auto-Seed:** Click the **Auto-Seed** button to automatically detect peaks using background subtraction and geodesic filtering.
    *   **CSV Import:** Click **CSV Seed 2** or **CSV Seed 3** to load external coordinate predictions (e.g., `*seeds_2-respan.csv`).
4. **Target Management:** 
    *   Right-click to remove erroneous seeds.
    *   Assign specific classifications to seeds using the **Spine (z)**, **Sub (c)**, or **Filo (x)** buttons.
    *   Update target tracking statuses (`static`, `new`, `eliminated`) using the Status dropdown.
    *   Manually refine the dendritic barrier by switching the mode to **Paint/Erase Barrier** to handle complex overlapping structures.
5. **Adjust Topological Restraints:** 
    *   To prevent the segmentation mask from bleeding down the dendrite shaft, ensure **Use 2D Curvature Isolation** is checked.
    *   Adjust the **Hessian σ** to match the macroscopic size of the spines (higher values for massive mushroom spines, lower for thin filopodia).
    *   Tune the **Blob Strictness** to control where the algorithm severs the spine neck.
6. **Batch Analysis:** Click **Analyze All Targets**. The tool uses a 2.5D Dilated Envelope Extrusion technique to accurately map 3D volume while physically preventing longitudinal shaft bleeding. Results are automatically saved to an `output_analysis` folder, including a detailed CSV report, a filtered `.tif`, the segmentation mask, and a labeled MIP overlay image.

## Examples
Example demonstration videos of the pre-processing and UI workflow are available directly in this repository.

## Future Improvements
There are numerous opportunities for further development, including expanding this 2.5D framework into simpler, fully automated tools based on this manual ground-truth extraction pipeline.
