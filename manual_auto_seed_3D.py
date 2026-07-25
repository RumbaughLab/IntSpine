import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tifffile import imread, imwrite
from skimage.filters import gaussian
from skimage.draw import disk
from scipy.ndimage import distance_transform_edt, label, binary_fill_holes, uniform_filter, maximum_position, binary_erosion, gaussian_filter
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ==========================================
# 1. CONFIGURATION & STATE
# ==========================================
dz, dy, dx = 0.3, 0.108, 0.108  
voxel_volume = dx * dy * dz

state = {
    'input_folder': '',
    'output_folder': '',
    'files': [],
    'current_idx': 0,
    
    'raw_stack': None,
    'base_smoothed_stack': None,
    'dist_field_3d': None,
    'dendrite_length_um': 0.0,
    'initial_shaft_barrier': None,
    'avg_initial_dendrite_intensity': 0.0,
    
    'z': 0,
    'click_x': None, 'click_y': None,       
    'target_x': None, 'target_y': None, 'target_z': None, 
    'mask': None,
    'shaft_barrier': None,
    
    'painted_barrier_2d': None,
    'erased_barrier_2d': None,
    'is_drawing': False,
    
    'saved_targets': [], 
    'target_counter': 1,
    'texts_ax1': [],
    'texts_ax2': [],
    'dots_ax1': [],
    'dots_ax2': []
}

# --- SWC FUNCTION ---
def process_swc_file(stack_shape, swc_file, raw_stack):
    swc_data = pd.read_csv(swc_file, sep=r'\s+', comment='#', header=None,
                           names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'])
    
    skeleton_2d = np.zeros((stack_shape[1], stack_shape[2]), dtype=bool)
    for _, row in swc_data.iterrows():
        y_idx = int(round(row['y']/dy))
        x_idx = int(round(row['x']/dx))
        if (0 <= y_idx < stack_shape[1] and 0 <= x_idx < stack_shape[2]):
            skeleton_2d[y_idx, x_idx] = True

    distance_field_2d = distance_transform_edt(~skeleton_2d, sampling=[dy, dx])
    dist_3d = np.broadcast_to(distance_field_2d, stack_shape).copy()
    
    swc_dict = swc_data.set_index('id').to_dict('index')
    total_length = 0.0
    for node_id, data in swc_dict.items():
        parent_id = data['parent']
        if parent_id in swc_dict:
            parent_data = swc_dict[parent_id]
            dist = np.sqrt((data['x'] - parent_data['x'])**2 + 
                           (data['y'] - parent_data['y'])**2 + 
                           (data['z'] - parent_data['z'])**2)
            total_length += dist
            
    return dist_3d, total_length

# ==========================================
# 2. SESSION STATE & UI SETUP
# ==========================================
pink_cmap = ListedColormap(['#ff69b4'])
green_cmap = ListedColormap(['#00ff00'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 5.5))
plt.subplots_adjust(bottom=0.05, top=0.88, left=0.02, right=0.98, wspace=0.05)
fig.suptitle("IntSpine: Interactive Spine Analysis", fontsize=14, fontweight='bold')

dummy_img = np.zeros((10, 10))

img_display = ax1.imshow(dummy_img, cmap='gray')
barrier_display = ax1.imshow(dummy_img, cmap=pink_cmap, alpha=0.3)
mask_display = ax1.imshow(dummy_img, cmap=green_cmap, alpha=0.6)
click_marker, = ax1.plot([], [], 'ro', markersize=4) 
target_marker, = ax1.plot([], [], 'c+', markersize=15, markeredgewidth=2) 

brush_circle_ax1 = plt.Circle((0, 0), 1, color='cyan', fill=False, linewidth=1.5, visible=False)
brush_circle_ax2 = plt.Circle((0, 0), 1, color='cyan', fill=False, linewidth=1.5, visible=False)
ax1.add_patch(brush_circle_ax1)
ax2.add_patch(brush_circle_ax2)

ax1.set_title("Z-Slice Navigator")

mip_display = ax2.imshow(dummy_img, cmap='gray')
mip_barrier_display = ax2.imshow(dummy_img, cmap=pink_cmap, alpha=0.2)
mip_mask_display = ax2.imshow(dummy_img, cmap=green_cmap, alpha=0.6)
click_marker_mip, = ax2.plot([], [], 'ro', markersize=4)
target_marker_mip, = ax2.plot([], [], 'c+', markersize=15, markeredgewidth=2)
ax2.set_title("Global MIP (L-Click: Select | R-Click: Remove)")

# --- Widgets ---
input_folder_input = widgets.Text(placeholder='C:/Path/To/Folder', description='Input Folder:', layout=widgets.Layout(width='360px'))
load_folder_btn = widgets.Button(description='Load Remaining', button_style='primary', icon='folder-open', layout=widgets.Layout(width='150px'))
file_info_label = widgets.Label(value="No folder loaded")

mask_source_dropdown = widgets.Dropdown(options=['SWC Mask', 'Geo Mask', 'Respan Mask'], value='SWC Mask', description='Mask Source:', layout=widgets.Layout(width='360px'))

mode_radio = widgets.RadioButtons(options=['Target Spines', 'Paint Barrier', 'Erase Barrier'], value='Target Spines', description='Mode:', layout=widgets.Layout(width='160px'))
brush_size_slider = widgets.IntSlider(value=8, min=2, max=40, description='Brush Size:', style={'description_width': '80px'}, layout=widgets.Layout(width='230px'))
clear_paint_btn = widgets.Button(description='Clear Paint', button_style='danger', icon='eraser', layout=widgets.Layout(width='110px'))
save_barrier_btn = widgets.Button(description='Save Barrier', button_style='info', icon='save', layout=widgets.Layout(width='115px'))

z_slider = widgets.IntSlider(value=0, min=0, max=1, description='Z-Slice:', style={'description_width': '90px'}, layout=widgets.Layout(width='360px'))
wl_slider = widgets.IntRangeSlider(value=[0, 1], min=0, max=1, description='Win/Lvl:', style={'description_width': '90px'}, layout=widgets.Layout(width='360px'))
barrier_slider = widgets.FloatSlider(value=1.2, min=0.0, max=4.0, step=0.1, description='Barrier µm:', style={'description_width': '90px'}, layout=widgets.Layout(width='360px'))
tol_slider = widgets.FloatSlider(value=0.45, min=0.05, max=0.90, step=0.05, description='Tolerance:', style={'description_width': '90px'}, layout=widgets.Layout(width='360px'))
z_search_slider = widgets.IntSlider(value=10, min=0, max=20, step=1, description='Z-Search:', style={'description_width': '90px'}, layout=widgets.Layout(width='360px'))
max_geodesic_slider = widgets.FloatSlider(value=5.0, min=1.0, max=15.0, step=0.5, description='Max Geodesic µm:', style={'description_width': '105px'}, layout=widgets.Layout(width='360px'))

# NEW: Hessian Curvature Isolation UI
use_hessian_cb = widgets.Checkbox(value=True, description='Use Curvature Isolation', indent=False, layout=widgets.Layout(width='230px'))
strictness_slider = widgets.FloatSlider(value=0.01, min=-0.05, max=0.05, step=0.005, description='Blob Strictness:', style={'description_width': '100px'}, layout=widgets.Layout(width='360px'))
hessian_sigma_slider = widgets.FloatSlider(value=1.0, min=0.5, max=4.0, step=0.1, description='Hessian \u03c3:', style={'description_width': '100px'}, layout=widgets.Layout(width='360px'))

show_targets_cb = widgets.Checkbox(value=True, description='Show Markers', indent=False, layout=widgets.Layout(width='140px'))
show_mask_cb = widgets.Checkbox(value=True, description='Show Segment', indent=False, layout=widgets.Layout(width='140px'))

target_list_ui = widgets.SelectMultiple(options=[], description='Target List:', style={'description_width': 'initial'}, layout={'width': '365px', 'height': '150px'})
delete_target_btn = widgets.Button(description='Delete Selected Targets', button_style='danger', icon='trash', layout=widgets.Layout(width='365px'))

# Status Application UI
status_dropdown = widgets.Dropdown(options=['static', 'new', 'eliminated'], value='static', description='Status:', layout=widgets.Layout(width='180px'))
apply_status_btn = widgets.Button(description='Apply Status', button_style='info', icon='check', layout=widgets.Layout(width='180px'))

# Rename UI
rename_id_input = widgets.Text(placeholder='New ID', layout=widgets.Layout(width='180px'))
rename_id_btn = widgets.Button(description='Update Target ID', button_style='warning', icon='edit', layout=widgets.Layout(width='180px'))

# Control Buttons
auto_seed_btn = widgets.Button(description='Auto-Seed', button_style='warning', icon='magic', layout=widgets.Layout(width='115px'))
respan_seed2_btn = widgets.Button(description='CSV Seed 2', button_style='info', icon='file-text', layout=widgets.Layout(width='115px'))
respan_seed3_btn = widgets.Button(description='CSV Seed 3', button_style='info', icon='file-text', layout=widgets.Layout(width='115px'))
seed_buttons_ui = widgets.HBox([auto_seed_btn, respan_seed2_btn, respan_seed3_btn])

save_target_btn = widgets.Button(description='Spine (z)', button_style='info', icon='bookmark', layout=widgets.Layout(width='115px'))
suboptimal_btn = widgets.Button(description='Sub (c)', button_style='danger', icon='exclamation-triangle', layout=widgets.Layout(width='115px'))
filopodia_btn = widgets.Button(description='Filo (x)', button_style='warning', icon='tag', layout=widgets.Layout(width='115px'))
undo_target_btn = widgets.Button(description='Undo Last', button_style='danger', icon='undo', layout=widgets.Layout(width='180px'))

reset_view_btn = widgets.Button(description='Reset View (a)', button_style='', icon='home', layout=widgets.Layout(width='180px'))
zoom_rect_btn = widgets.Button(description='Zoom Rect (f)', button_style='', icon='search', layout=widgets.Layout(width='180px'))
pan_btn = widgets.Button(description='Pan Image (d)', button_style='', icon='arrows', layout=widgets.Layout(width='180px'))

analyze_all_btn = widgets.Button(description='Analyze All Targets', button_style='success', icon='cogs', layout=widgets.Layout(width='365px'))
next_image_btn = widgets.Button(description='Next Image >', button_style='primary', icon='arrow-right', layout=widgets.Layout(width='365px'))

custom_id_input = widgets.Text(placeholder='Override Start ID...', description='Custom ID:', layout=widgets.Layout(width='365px'))

log_output = widgets.Output()

# ==========================================
# 3. INTERACTIVE CALLBACKS & SHARED LOGIC
# ==========================================
def get_effective_barrier():
    base = state['shaft_barrier'] if state['shaft_barrier'] is not None else np.zeros_like(state['raw_stack'], dtype=bool)
    painted_3d = np.broadcast_to(state['painted_barrier_2d'], base.shape) if state['painted_barrier_2d'] is not None else np.zeros_like(base)
    erased_3d = np.broadcast_to(state['erased_barrier_2d'], base.shape) if state['erased_barrier_2d'] is not None else np.zeros_like(base)
    
    effective = (base | painted_3d) & (~erased_3d)
    return effective

def get_hessian_blob_mask(V_sub, strictness, sig=1.0):
    """Computes the 3D Hessian locally and uses the largest eigenvalue to sever shafts/necks."""
    if V_sub.shape[0] >= 3:
        Dzz = gaussian_filter(V_sub, sigma=sig, order=[2, 0, 0])
        Dyy = gaussian_filter(V_sub, sigma=sig, order=[0, 2, 0])
        Dxx = gaussian_filter(V_sub, sigma=sig, order=[0, 0, 2])
        Dzy = gaussian_filter(V_sub, sigma=sig, order=[1, 1, 0])
        Dzx = gaussian_filter(V_sub, sigma=sig, order=[1, 0, 1])
        Dyx = gaussian_filter(V_sub, sigma=sig, order=[0, 1, 1])

        H = np.zeros((3, 3) + V_sub.shape)
        H[0, 0] = Dzz; H[0, 1] = Dzy; H[0, 2] = Dzx
        H[1, 0] = Dzy; H[1, 1] = Dyy; H[1, 2] = Dyx
        H[2, 0] = Dzx; H[2, 1] = Dyx; H[2, 2] = Dxx
        H = np.moveaxis(H, [0, 1], [-2, -1])

        l1 = np.linalg.eigvalsh(H)[..., 2] 
    else:
        l1 = np.zeros_like(V_sub)
        for zi in range(V_sub.shape[0]):
            slice_v = V_sub[zi]
            Dyy = gaussian_filter(slice_v, sigma=sig, order=[2, 0])
            Dxx = gaussian_filter(slice_v, sigma=sig, order=[0, 2])
            Dxy = gaussian_filter(slice_v, sigma=sig, order=[1, 1])
            H2 = np.zeros((2, 2) + slice_v.shape)
            H2[0,0]=Dyy; H2[0,1]=Dxy; H2[1,0]=Dxy; H2[1,1]=Dxx
            H2 = np.moveaxis(H2, [0,1], [-2,-1])
            l1[zi] = np.linalg.eigvalsh(H2)[..., 1]
            
    return l1 <= strictness

def apply_mask_source():
    if state['raw_stack'] is None: return
    
    filepath = state['files'][state['current_idx']]
    dir_name = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
    
    val = mask_source_dropdown.value
    raw_stack = state['raw_stack']
    
    dist_3d = np.inf * np.ones_like(raw_stack)
    d_len = 0.0
    
    with log_output:
        if val == 'SWC Mask':
            swc_path = filepath.replace('.tif', '.swc').replace('.tiff', '.swc')
            if os.path.exists(swc_path):
                dist_3d, d_len = process_swc_file(raw_stack.shape, swc_path, raw_stack)
                print(f"✅ Found SWC! Length: {d_len:.2f} µm")
            else:
                print("⚠️ No SWC file found. Default barrier disabled.")
        else:
            pattern = "*geo*.tif" if val == 'Geo Mask' else "*respan*.tif"
            search_path = os.path.join(dir_name, f"{base_name}{pattern}")
            matches = glob(search_path)
            if not matches:
                search_path2 = os.path.join(dir_name, f"*{pattern}")
                matches = glob(search_path2)
                
            if matches:
                mask_file = matches[0]
                print(f"✅ Found {val} file: {os.path.basename(mask_file)}")
                loaded_mask = imread(mask_file) > 0
                
                if loaded_mask.ndim == 2:
                    dist_2d = distance_transform_edt(~loaded_mask, sampling=[dy, dx])
                    dist_3d = np.broadcast_to(dist_2d, raw_stack.shape).copy()
                elif loaded_mask.ndim == 3:
                    if loaded_mask.shape != raw_stack.shape:
                        print(f"⚠️ Mask shape {loaded_mask.shape} mismatches image {raw_stack.shape}. Attempting direct processing.")
                    dist_3d = distance_transform_edt(~loaded_mask, sampling=[dz, dy, dx])
                
                print("💡 Tip: Set 'Barrier µm' slider to 0.0 if you want to use the exact pre-calculated mask without dilation.")
            else:
                print(f"⚠️ No {val} file found matching '{pattern}'. Barrier disabled.")
                
    state['dist_field_3d'] = dist_3d
    state['dendrite_length_um'] = d_len
    state['initial_shaft_barrier'] = state['dist_field_3d'] <= barrier_slider.value
    
    if np.any(state['initial_shaft_barrier']):
        state['avg_initial_dendrite_intensity'] = float(np.mean(raw_stack[state['initial_shaft_barrier']]))
    else:
        state['avg_initial_dendrite_intensity'] = 0.0
        
    state['shaft_barrier'] = state['dist_field_3d'] <= barrier_slider.value

def on_mask_source_change(change):
    val = change['new']
    if val in ['Geo Mask', 'Respan Mask']:
        barrier_slider.value = 0.0
        
    if state['raw_stack'] is not None:
        apply_mask_source()
        refresh_display()

def find_optimal_xyz(x, y, search_radius=5):
    """
    Unified function utilizing Distance-Penalized Peak Finding (Strategy A).
    It multiplies voxel intensity by geodesic distance to pull seeds strictly away from the bright shaft halo.
    """
    if state['base_smoothed_stack'] is None: 
        return state['z'], y, x
        
    stack = state['base_smoothed_stack']
    
    y_min = max(0, y - search_radius)
    y_max = min(stack.shape[1], y + search_radius + 1)
    x_min = max(0, x - search_radius)
    x_max = min(stack.shape[2], x + search_radius + 1)
    
    sub_volume = stack[:, y_min:y_max, x_min:x_max].copy().astype(np.float64)
    sub_volume = uniform_filter(sub_volume, size=3)
    
    dist_sub = state['dist_field_3d'][:, y_min:y_max, x_min:x_max]
    dist_weight = np.where(np.isinf(dist_sub), 1.0, dist_sub)
    score_volume = sub_volume * dist_weight
    
    eff_barrier_3d = get_effective_barrier()
    eff_bar_sub = eff_barrier_3d[:, y_min:y_max, x_min:x_max]
    
    score_volume[eff_bar_sub] = -1e9
            
    if np.all(score_volume == -1e9):
        return state['z'], y, x 
    
    max_idx = np.argmax(score_volume)
    z_loc, dy_loc, dx_loc = np.unravel_index(max_idx, score_volume.shape)
    
    opt_z = int(z_loc)
    opt_y = int(y_min + dy_loc)
    opt_x = int(x_min + dx_loc)
    
    return opt_z, opt_y, opt_x

def auto_generate_seeds(b=None):
    if state['raw_stack'] is None: return
    
    with log_output:
        clear_output()
        print("🔍 Running background subtraction & geodesic filtering for auto-seeding...")
        
    state['saved_targets'] = []
    
    custom_val = custom_id_input.value.strip()
    if custom_val.isdigit():
        state['target_counter'] = int(custom_val)
        custom_id_input.value = ''
    else:
        state['target_counter'] = 1
        
    target_list_ui.options = []
        
    mip_img = np.max(state['raw_stack'], axis=0)
    
    background = uniform_filter(mip_img.astype(float), size=50)
    bg_subtracted = np.clip(mip_img.astype(float) - background, 0, None)
    
    smoothed = gaussian(bg_subtracted, sigma=1.5)
    
    if smoothed.max() > 0:
        thresh_val = np.percentile(smoothed[smoothed > 0], 88)
        binary_objects = smoothed > thresh_val
        
        eff_barrier_3d = get_effective_barrier()
        eff_barrier_2d = np.max(eff_barrier_3d, axis=0)
        binary_objects[eff_barrier_2d] = False
        
        clean_objects = binary_fill_holes(binary_objects)
        labeled_objects, num_features = label(clean_objects)
        
        dist_field_2d = state['dist_field_3d'][0] if state['dist_field_3d'] is not None else np.zeros_like(mip_img, dtype=float)
        max_geo_dist = max_geodesic_slider.value
        
        new_targets = []
        filtered_count = 0
        for i in range(1, num_features + 1):
            if np.sum(labeled_objects == i) < 5: continue
            
            y_loc, x_loc = maximum_position(mip_img, labeled_objects, i)
            y_loc, x_loc = int(y_loc), int(x_loc)
            
            geo_dist = dist_field_2d[y_loc, x_loc]
            if geo_dist > max_geo_dist:
                filtered_count += 1
                continue
            
            opt_z, opt_y, opt_x = find_optimal_xyz(x_loc, y_loc, search_radius=5)
            
            idx = state['target_counter']
            status = 'static'
            label_text = f"[{idx}] Z:{opt_z+1} Y:{opt_y} X:{opt_x} ({status})"
            new_targets.append({
                'idx': idx, 'label': label_text, 
                'z': opt_z, 'y': opt_y, 'x': opt_x,
                'click_x': x_loc, 'click_y': y_loc,
                'target_type': 'spine',
                'status': status
            })
            state['target_counter'] += 1
            
        state['saved_targets'].extend(new_targets)
        target_list_ui.options = [t['label'] for t in state['saved_targets']]
        
        with log_output:
            print(f"✅ Extracted {len(new_targets)} seeds (Filtered out {filtered_count} objects > {max_geo_dist} µm away)!")
            
    refresh_display()

def load_csv_seeds(seed_type):
    if state['raw_stack'] is None: return
    
    filepath = state['files'][state['current_idx']]
    dir_name = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
    
    suffix = "seeds_2-respan" if seed_type == 'seed2' else "seeds_3-respan"
    
    search_path = os.path.join(dir_name, f"{base_name}*{suffix}*.csv")
    matches = glob(search_path)
    if not matches:
        search_path2 = os.path.join(dir_name, f"*{suffix}*.csv")
        matches = glob(search_path2)
        
    with log_output:
        clear_output()
        if not matches:
            print(f"⚠️ No CSV file found matching '*{suffix}*.csv' in the folder.")
            return
            
        csv_file = matches[0]
        print(f"✅ Found CSV file: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file)
            x_col = next((c for c in df.columns if c.lower() in ['x', 'corrected_x', 'centroid_x']), None)
            y_col = next((c for c in df.columns if c.lower() in ['y', 'corrected_y', 'centroid_y']), None)
            
            if x_col is None or y_col is None:
                print("❌ Could not find an 'x' and 'y' column in the loaded CSV.")
                return
                
            state['saved_targets'] = []
            
            custom_val = custom_id_input.value.strip()
            if custom_val.isdigit():
                state['target_counter'] = int(custom_val)
                custom_id_input.value = ''
            else:
                state['target_counter'] = 1
                
            target_list_ui.options = []
            
            loaded_count = 0
            for _, row in df.iterrows():
                x_loc = int(round(row[x_col]))
                y_loc = int(round(row[y_col]))
                
                if not (0 <= y_loc < state['raw_stack'].shape[1] and 0 <= x_loc < state['raw_stack'].shape[2]):
                    continue
                    
                opt_z, opt_y, opt_x = find_optimal_xyz(x_loc, y_loc, search_radius=5)
                
                idx = state['target_counter']
                status = 'static'
                label_text = f"[{idx}] Z:{opt_z+1} Y:{opt_y} X:{opt_x} ({status})"
                
                state['saved_targets'].append({
                    'idx': idx, 'label': label_text, 
                    'z': opt_z, 'y': opt_y, 'x': opt_x,
                    'click_x': x_loc, 'click_y': y_loc,
                    'target_type': 'spine',
                    'status': status
                })
                state['target_counter'] += 1
                loaded_count += 1
                
            target_list_ui.options = [t['label'] for t in state['saved_targets']]
            print(f"✅ Successfully loaded and snapped {loaded_count} targets from {os.path.basename(csv_file)}!")
            
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            
    refresh_display()

def load_image_at_index(idx):
    if not state['files'] or idx >= len(state['files']): return
    
    filepath = state['files'][idx]
    filename = os.path.basename(filepath)
    file_info_label.value = f"Remaining {idx+1}/{len(state['files'])}: {filename}"
    
    in_path = input_folder_input.value.strip()
    if not in_path: in_path = os.path.dirname(filepath)
    out_dir = os.path.join(in_path, 'output_analysis')
    base_name = os.path.splitext(filename)[0]
    if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
    
    custom_barrier_2d_path = os.path.join(out_dir, base_name + '_custom_barrier_2d.tif')
    
    with log_output:
        clear_output()
        print(f"Loading {filename}...")
        
    raw_stack = imread(filepath)
    state['raw_stack'] = raw_stack
    state['base_smoothed_stack'] = gaussian(raw_stack, sigma=1.0, preserve_range=True).astype(raw_stack.dtype)
    
    h, w = raw_stack.shape[1], raw_stack.shape[2]
    ax1.set_xlim(-0.5, w - 0.5)
    ax1.set_ylim(h - 0.5, -0.5)
    ax2.set_xlim(-0.5, w - 0.5)
    ax2.set_ylim(h - 0.5, -0.5)
    
    extent = [-0.5, w - 0.5, h - 0.5, -0.5]
    for disp in [img_display, barrier_display, mask_display, mip_display, mip_barrier_display, mip_mask_display]:
        disp.set_extent(extent)
    
    apply_mask_source()

    if os.path.exists(custom_barrier_2d_path):
        edit_mask = imread(custom_barrier_2d_path)
        state['painted_barrier_2d'] = (edit_mask == 1)
        state['erased_barrier_2d'] = (edit_mask == 2)
        with log_output: print("📂 Loaded previously saved custom 2D barrier edits!")
    else:
        state['painted_barrier_2d'] = np.zeros((h, w), dtype=bool)
        state['erased_barrier_2d'] = np.zeros((h, w), dtype=bool)

    state['z'] = 0
    state['mask'] = np.zeros_like(raw_stack, dtype=bool)
    state['is_drawing'] = False
    
    state['click_x'] = state['click_y'] = None
    state['target_x'] = state['target_y'] = state['target_z'] = None
    state['saved_targets'] = []
    state['target_counter'] = 1
    custom_id_input.value = ''
    target_list_ui.options = []
    
    z_slider.unobserve_all()
    z_slider.max = raw_stack.shape[0] - 1
    z_slider.value = 0
    z_slider.observe(lambda c: state.update({'z': c['new']}) or refresh_display(), names='value')
    
    wl_slider.min = raw_stack.min()
    wl_slider.max = raw_stack.max()
    wl_slider.value = [np.percentile(raw_stack, 1), np.percentile(raw_stack, 99)]
    
    mip_raw = np.max(raw_stack, axis=0)
    mip_display.set_data(mip_raw)
    mip_display.set_clim(wl_slider.value[0], wl_slider.value[1])
    
    refresh_display()

def refresh_display():
    if state['raw_stack'] is None: return
    z = state['z']
    
    img_display.set_data(state['raw_stack'][z])
    img_display.set_clim(wl_slider.value[0], wl_slider.value[1])
    
    eff_barrier = get_effective_barrier()
    barrier_z = eff_barrier[z]
    barrier_mip = np.max(eff_barrier, axis=0)
    
    barrier_display.set_data(np.ma.masked_where(~barrier_z, barrier_z))
    mip_barrier_display.set_data(np.ma.masked_where(~barrier_mip, barrier_mip))
    
    if show_mask_cb.value:
        mask_display.set_data(np.ma.masked_where(~state['mask'][z], state['mask'][z]))
        mip_m = np.max(state['mask'], axis=0)
        mip_mask_display.set_data(np.ma.masked_where(~mip_m, mip_m))
    else:
        mask_display.set_data(np.ma.masked_all(state['raw_stack'][z].shape))
        mip_mask_display.set_data(np.ma.masked_all(barrier_mip.shape))
    
    for item in state['texts_ax1'] + state['texts_ax2'] + state['dots_ax1'] + state['dots_ax2']:
        try: item.remove()
        except: pass
    state['texts_ax1'].clear(); state['texts_ax2'].clear()
    state['dots_ax1'].clear(); state['dots_ax2'].clear()
    
    if show_targets_cb.value:
        if state['click_x'] is not None and mode_radio.value == 'Target Spines':
            click_marker.set_data([state['click_x']], [state['click_y']])
            click_marker_mip.set_data([state['click_x']], [state['click_y']])
        else:
            click_marker.set_data([], [])
            click_marker_mip.set_data([], [])

        if state['target_z'] is not None and state['target_x'] is not None and mode_radio.value == 'Target Spines':
            target_marker.set_data([state['target_x']], [state['target_y']])
            target_marker_mip.set_data([state['target_x']], [state['target_y']])
        else:
            target_marker.set_data([], [])
            target_marker_mip.set_data([], [])
            
        if state['saved_targets']:
            for t in state['saved_targets']:
                t_type = t.get('target_type', 'spine')
                if t_type == 'filopodia':
                    color = 'blue'
                elif t_type == 'suboptimal':
                    color = 'orange'
                else:
                    color = 'red'
                
                cx, cy = t.get('click_x', t['x']), t.get('click_y', t['y'])
                d1_c, = ax1.plot(cx, cy, marker='o', color='red', markersize=3, linestyle='None')
                d2_c, = ax2.plot(cx, cy, marker='o', color='red', markersize=3, linestyle='None')
                state['dots_ax1'].append(d1_c)
                state['dots_ax2'].append(d2_c)
                
                txt1 = ax1.text(cx + 3, cy, str(t['idx']), color=color, fontsize=10, fontweight='bold')
                txt2 = ax2.text(cx + 3, cy, str(t['idx']), color=color, fontsize=10, fontweight='bold')
                state['texts_ax1'].append(txt1)
                state['texts_ax2'].append(txt2)
    else:
        click_marker.set_data([], [])
        click_marker_mip.set_data([], [])
        target_marker.set_data([], [])
        target_marker_mip.set_data([], [])
    
    z_slider.value = z
    fig.canvas.draw_idle()

def paint(event):
    if not state['is_drawing'] or state['raw_stack'] is None: return
    if event.inaxes not in [ax1, ax2]: return
    
    x, y = int(round(event.xdata)), int(round(event.ydata))
    rr, cc = disk((y, x), brush_size_slider.value, shape=state['painted_barrier_2d'].shape)
    
    if mode_radio.value == 'Paint Barrier':
        state['painted_barrier_2d'][rr, cc] = True
        state['erased_barrier_2d'][rr, cc] = False
    elif mode_radio.value == 'Erase Barrier':
        state['erased_barrier_2d'][rr, cc] = True
        state['painted_barrier_2d'][rr, cc] = False
    
    z = state['z']
    eff_barrier = get_effective_barrier()
    barrier_display.set_data(np.ma.masked_where(~eff_barrier[z], eff_barrier[z]))
    mip_barrier_display.set_data(np.ma.masked_where(~np.max(eff_barrier, axis=0), np.max(eff_barrier, axis=0)))
    fig.canvas.draw_idle()

def on_mouse_press(event):
    if fig.canvas.toolbar.mode != '': return
    
    if mode_radio.value in ['Paint Barrier', 'Erase Barrier']:
        state['is_drawing'] = True
        paint(event)
        
    elif mode_radio.value == 'Target Spines' and event.inaxes in [ax1, ax2]:
        clicked_x, clicked_y = int(round(event.xdata)), int(round(event.ydata))
        
        if event.button == 3:
            min_dist = 12
            to_remove = None
            for t in state['saved_targets']:
                dist = np.hypot(t['click_x'] - clicked_x, t['click_y'] - clicked_y)
                if dist < min_dist:
                    min_dist = dist
                    to_remove = t
                    
            if to_remove:
                state['saved_targets'].remove(to_remove)
                target_list_ui.options = [t['label'] for t in state['saved_targets']]
                refresh_display()
                with log_output:
                    clear_output()
                    print(f"🗑️ Removed Target [{to_remove['idx']}] at X:{to_remove['click_x']}, Y:{to_remove['click_y']}")
            return
            
        if event.button == 1:
            state['click_x'] = clicked_x
            state['click_y'] = clicked_y
            
            opt_z, opt_y, opt_x = find_optimal_xyz(clicked_x, clicked_y, search_radius=5)
            
            state['target_z'] = opt_z
            state['target_y'] = opt_y
            state['target_x'] = opt_x
            state['z'] = opt_z
            
            refresh_display()

def on_mouse_motion(event):
    if event.inaxes in [ax1, ax2] and event.xdata is not None and event.ydata is not None:
        r = brush_size_slider.value
        if mode_radio.value in ['Paint Barrier', 'Erase Barrier']:
            brush_circle_ax1.center = (event.xdata, event.ydata)
            brush_circle_ax1.set_radius(r)
            brush_circle_ax1.set_visible(event.inaxes == ax1)
            
            brush_circle_ax2.center = (event.xdata, event.ydata)
            brush_circle_ax2.set_radius(r)
            brush_circle_ax2.set_visible(event.inaxes == ax2)
            fig.canvas.draw_idle()
        else:
            brush_circle_ax1.set_visible(False)
            brush_circle_ax2.set_visible(False)
            
    paint(event)

def on_mouse_release(event):
    state['is_drawing'] = False

def on_clear_paint(b):
    if state['raw_stack'] is not None:
        state['painted_barrier_2d'] = np.zeros_like(state['painted_barrier_2d'])
        state['erased_barrier_2d'] = np.zeros_like(state['erased_barrier_2d'])
        refresh_display()

def on_save_barrier(b=None):
    if state['raw_stack'] is None: return
    
    in_path = input_folder_input.value.strip()
    if not in_path: in_path = os.path.dirname(state['files'][state['current_idx']])
    
    out_dir = os.path.join(in_path, 'output_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    current_file = state['files'][state['current_idx']]
    base_name = os.path.splitext(os.path.basename(current_file))[0]
    if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        
    barrier_img_path = os.path.join(out_dir, base_name + '_custom_barrier_2d.tif')
    
    h, w = state['raw_stack'].shape[1], state['raw_stack'].shape[2]
    edit_mask = np.zeros((h, w), dtype=np.uint8)
    edit_mask[state['painted_barrier_2d']] = 1
    edit_mask[state['erased_barrier_2d']] = 2
    
    imwrite(barrier_img_path, edit_mask)
    
    with log_output:
        print(f"💾 Custom 2D barrier edits saved successfully to {base_name}_custom_barrier_2d.tif")

def on_barrier_change(change):
    if state['dist_field_3d'] is not None:
        state['shaft_barrier'] = state['dist_field_3d'] <= change['new']
        refresh_display()

def on_reset_view(b=None):
    if fig.canvas.toolbar.mode == 'zoom rect':
        fig.canvas.toolbar.zoom()
    elif fig.canvas.toolbar.mode == 'pan/zoom':
        fig.canvas.toolbar.pan()
        
    if state['raw_stack'] is not None:
        h, w = state['raw_stack'].shape[1], state['raw_stack'].shape[2]
        ax1.set_xlim(-0.5, w - 0.5)
        ax1.set_ylim(h - 0.5, -0.5)
        ax2.set_xlim(-0.5, w - 0.5)
        ax2.set_ylim(h - 0.5, -0.5)
        fig.canvas.draw_idle()

def on_zoom_rect(b=None):
    if fig.canvas.toolbar.mode == 'pan/zoom':
        fig.canvas.toolbar.pan()
    fig.canvas.toolbar.zoom()

def on_pan(b=None):
    if fig.canvas.toolbar.mode == 'zoom rect':
        fig.canvas.toolbar.zoom()
    fig.canvas.toolbar.pan()

def on_scroll(event):
    if event.inaxes not in [ax1, ax2] or state['raw_stack'] is None: return
    if event.button == 'up':
        state['z'] = min(state['z'] + 1, state['raw_stack'].shape[0] - 1)
    elif event.button == 'down':
        state['z'] = max(state['z'] - 1, 0)
    refresh_display()

def on_save_target(target_type='spine'):
    if state['click_x'] is not None and mode_radio.value == 'Target Spines':
        opt_z, opt_y, opt_x = state['target_z'], state['target_y'], state['target_x']
        click_x, click_y = state['click_x'], state['click_y']
        
        custom_val = custom_id_input.value.strip()
        if custom_val.isdigit():
            state['target_counter'] = int(custom_val)
            custom_id_input.value = ''
            
        idx = state['target_counter']
        
        if target_type == 'filopodia':
            label_prefix = "[Filo]"
        elif target_type == 'suboptimal':
            label_prefix = "[Sub]"
        else:
            label_prefix = ""
            
        status = 'static'
        label_text = f"{label_prefix} [{idx}] Z:{opt_z+1} Y:{opt_y} X:{opt_x} ({status})".strip()
        
        if not any(t['z'] == opt_z and t['y'] == opt_y and t['x'] == opt_x for t in state['saved_targets']):
            state['saved_targets'].append({
                'idx': idx, 'label': label_text, 
                'z': opt_z, 'y': opt_y, 'x': opt_x,
                'click_x': click_x, 'click_y': click_y,
                'target_type': target_type,
                'status': status
            })
            state['target_counter'] += 1
            target_list_ui.options = [t['label'] for t in state['saved_targets']]
            
            with log_output:
                clear_output()
                if target_type == 'filopodia': tag = "Filopodia"
                elif target_type == 'suboptimal': tag = "Suboptimal Spine"
                else: tag = "Target Spine"
                print(f"💾 Saved {tag} {idx} at Z:{opt_z+1}, Y:{opt_y}, X:{opt_x} ({status})")
            
            refresh_display()

def on_delete_selected_target(b=None):
    selected_labels = target_list_ui.value
    if selected_labels and state['saved_targets']:
        for sel_label in selected_labels:
            target_data = next((t for t in state['saved_targets'] if t['label'] == sel_label), None)
            if target_data:
                state['saved_targets'].remove(target_data)
                
        target_list_ui.options = [t['label'] for t in state['saved_targets']]
        refresh_display()
        with log_output:
            clear_output()
            print(f"🗑️ Deleted {len(selected_labels)} Target(s) from list.")

def on_rename_target(b=None):
    selected_labels = target_list_ui.value
    new_id_str = rename_id_input.value.strip()
    
    if not selected_labels or len(selected_labels) > 1:
        with log_output:
            clear_output()
            print("⚠️ Please select exactly ONE target from the list to update its ID.")
        return
        
    selected_label = selected_labels[0]
    
    if not new_id_str.isdigit():
        with log_output:
            clear_output()
            print("⚠️ Please enter a valid number for the new ID.")
        return
        
    new_id = int(new_id_str)
    target_data = next((t for t in state['saved_targets'] if t['label'] == selected_label), None)
    
    if target_data:
        target_data['idx'] = new_id
        
        t_type = target_data.get('target_type', 'spine')
        label_prefix = ""
        if t_type == 'filopodia': label_prefix = "[Filo]"
        elif t_type == 'suboptimal': label_prefix = "[Sub]"
            
        status = target_data.get('status', 'static')
        new_label = f"{label_prefix} [{new_id}] Z:{target_data['z']+1} Y:{target_data['y']} X:{target_data['x']} ({status})".strip()
        
        target_data['label'] = new_label
        target_list_ui.options = [t['label'] for t in state['saved_targets']]
        target_list_ui.value = (new_label,)
        rename_id_input.value = ''
        
        refresh_display()
        with log_output:
            clear_output()
            print(f"✏️ Updated target to ID [{new_id}]")

def on_apply_status(b=None):
    selected_labels = target_list_ui.value
    new_status = status_dropdown.value
    
    if not selected_labels:
        with log_output:
            clear_output()
            print("⚠️ Please select at least one target to apply status.")
        return
        
    for sel_label in selected_labels:
        target_data = next((t for t in state['saved_targets'] if t['label'] == sel_label), None)
        if target_data:
            target_data['status'] = new_status
            
            t_type = target_data.get('target_type', 'spine')
            label_prefix = ""
            if t_type == 'filopodia': label_prefix = "[Filo]"
            elif t_type == 'suboptimal': label_prefix = "[Sub]"
                
            new_label = f"{label_prefix} [{target_data['idx']}] Z:{target_data['z']+1} Y:{target_data['y']} X:{target_data['x']} ({new_status})".strip()
            target_data['label'] = new_label
            
    updated_labels = [t['label'] for t in state['saved_targets']]
    target_list_ui.options = updated_labels
    
    refresh_display()
    with log_output:
        clear_output()
        print(f"✅ Applied status '{new_status}' to {len(selected_labels)} target(s).")

def on_undo_target(b=None):
    if state['saved_targets']:
        removed = state['saved_targets'].pop()
        target_list_ui.options = [t['label'] for t in state['saved_targets']]
        refresh_display()
        with log_output:
            clear_output()
            print(f"↩️ Undid Target [{removed['idx']}]. Remaining in queue: {len(state['saved_targets'])}")
    else:
        with log_output:
            clear_output()
            print("⚠️ Queue is already empty.")

def on_key_press(event):
    if event.key == 'z':
        on_save_target(target_type='spine')
    elif event.key == 'x':
        on_save_target(target_type='filopodia')
    elif event.key == 'c':
        on_save_target(target_type='suboptimal')
    elif event.key == 'u':
        on_undo_target()
    elif event.key in ['delete', 'backspace']:
        on_delete_selected_target()
    elif event.key == 'a':
        on_reset_view()
    elif event.key == 'f':
        on_zoom_rect()
    elif event.key == 'd':
        on_pan()
    elif event.key == 'escape':
        if fig.canvas.toolbar.mode == 'zoom rect':
            fig.canvas.toolbar.zoom()
        elif fig.canvas.toolbar.mode == 'pan/zoom':
            fig.canvas.toolbar.pan()

def on_target_selected(change):
    if change['new']:
        selected_label = change['new'][0]
        target_data = next((t for t in state['saved_targets'] if t['label'] == selected_label), None)
        if target_data:
            state['target_z'], state['target_y'], state['target_x'] = target_data['z'], target_data['y'], target_data['x']
            state['z'] = target_data['z']
            mode_radio.value = 'Target Spines'
            refresh_display()

def on_analyze_all(b):
    if not state['saved_targets'] or state['raw_stack'] is None:
        with log_output: clear_output(); print("⚠️ Queue is empty. Save targets first.")
        return
        
    on_save_barrier()
    
    combined_mask = np.zeros_like(state['raw_stack'], dtype=bool)
    results_list = []
    
    total_barrier = get_effective_barrier()
    current_smoothed_stack = np.where(total_barrier, 0, state['base_smoothed_stack'])
    
    current_barrier_val = barrier_slider.value
    current_tolerance_val = tol_slider.value
    current_zsearch_val = z_search_slider.value
    
    with log_output:
        print(f"⚙️ Batch processing targets...")
        
        for target in state['saved_targets']:
            z, y, x, idx = target['z'], target['y'], target['x'], target['idx']
            orig_x, orig_y = target['click_x'], target['click_y']
            
            t_type = target.get('target_type', 'spine')
            if target.get('is_filopodia', False): t_type = 'filopodia'
            
            target_status = target.get('status', 'static')
            
            geo_dist_um = float(state['dist_field_3d'][z, y, x]) if state['dist_field_3d'] is not None else 0.0
            if np.isinf(geo_dist_um): geo_dist_um = 0.0
            
            if t_type == 'filopodia':
                results_list.append({
                    'Target_ID': idx, 
                    'Classification': 'Filopodia',
                    'Status': target_status,
                    'Z_Slice': z + 1, 
                    'Original_Y': orig_y,
                    'Original_X': orig_x,
                    'Corrected_Y': y, 
                    'Corrected_X': x, 
                    'Local_Dendrite_Surface_Max': 0,
                    'Local_Dendrite_Surface_IntDen': 0.0,
                    'Area_Opt_Z_um2': 0.0,
                    'Geodesic_Distance_um': geo_dist_um,
                    'Vol_voxels': 0, 
                    'Vol_um3': 0.0,
                    'Max_Intensity': int(state['raw_stack'][z, y, x]),
                    'Sum_Intensity': 0.0,
                    'Integrated_Density': 0.0,
                    'Z_Slices_Count': 0,
                    'Avg_Initial_Dendrite_Intensity': state['avg_initial_dendrite_intensity'],
                    'Barrier_um': current_barrier_val,
                    'Tolerance': current_tolerance_val,
                    'Z_Search_Range': current_zsearch_val,
                    'Dendrite_Length_um': state['dendrite_length_um']
                })
                continue
                
            pad_xy = 30
            pad_z = current_zsearch_val
            z_min_loc = max(0, z - pad_z)
            z_max_loc = min(state['raw_stack'].shape[0], z + pad_z + 1)
            y_min_loc = max(0, y - pad_xy)
            y_max_loc = min(state['raw_stack'].shape[1], y + pad_xy + 1)
            x_min_loc = max(0, x - pad_xy)
            x_max_loc = min(state['raw_stack'].shape[2], x + pad_xy + 1)
            
            lz, ly, lx = z - z_min_loc, y - y_min_loc, x - x_min_loc

            if state['erased_barrier_2d'] is not None and state['erased_barrier_2d'][y, x]:
                seed_val = state['base_smoothed_stack'][z, y, x]
            else:
                seed_val = current_smoothed_stack[z, y, x]

            if seed_val == 0:
                print(f"❌ Target [{idx}] skipped: Seed is inside the pink dendritic barrier.")
                continue
                
            sub_stack = current_smoothed_stack[z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc]
            lower_bound = max(seed_val * (1.0 - current_tolerance_val), 1e-6)
            local_binary = sub_stack >= lower_bound
            
            # --- Strategy B: Hessian Curvature Isolation ---
            if use_hessian_cb.value:
                V = sub_stack.astype(float)
                v_max = V.max()
                if v_max > 0: V = V / v_max
                
                # Get map of geometric blobs (excluding tubes and saddles) using dynamic sigma
                blob_mask = get_hessian_blob_mask(V, strictness_slider.value, sig=hessian_sigma_slider.value)
                blob_mask[lz, ly, lx] = True # Protect initial anchor
                
                local_binary = local_binary & blob_mask
            
            labeled_local, _ = label(local_binary)
            seed_label = labeled_local[lz, ly, lx]
            
            if seed_label == 0:
                print(f"❌ Target [{idx}] skipped: Threshold too strict or point invalid.")
                continue
                
            local_spine_mask = (labeled_local == seed_label)
            local_spine_mask = binary_fill_holes(local_spine_mask)
            
            global_target_mask = np.zeros_like(combined_mask)
            global_target_mask[z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc] = local_spine_mask
            
            classification_label = 'Spine'
            
            if t_type == 'spine':
                combined_mask = np.logical_or(combined_mask, global_target_mask)
            elif t_type == 'suboptimal':
                classification_label = 'Suboptimal Measures'
            
            voxels = np.sum(global_target_mask)
            vol = voxels * voxel_volume
            max_intensity = int(state['raw_stack'][global_target_mask].max())
            sum_intensity = np.sum(state['raw_stack'][global_target_mask], dtype=np.float64)
            int_density = sum_intensity
            z_slices_count = int(np.sum(np.any(global_target_mask, axis=(1, 2))))
            
            area_opt_z_voxels = np.sum(global_target_mask[z, :, :])
            area_opt_z_um2 = area_opt_z_voxels * (dx * dy)

            local_barrier = total_barrier[z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc]
            local_raw = state['raw_stack'][z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc]
            
            local_barrier_surface = local_barrier & ~binary_erosion(local_barrier)
            
            if np.any(local_barrier_surface):
                local_dend_surf_max = int(local_raw[local_barrier_surface].max())
                local_dend_surf_sum = float(np.sum(local_raw[local_barrier_surface]))
            else:
                local_dend_surf_max = 0
                local_dend_surf_sum = 0.0
            
            results_list.append({
                'Target_ID': idx, 
                'Classification': classification_label,
                'Status': target_status,
                'Z_Slice': z + 1, 
                'Original_Y': orig_y,
                'Original_X': orig_x,
                'Corrected_Y': y, 
                'Corrected_X': x, 
                'Local_Dendrite_Surface_Max': local_dend_surf_max,
                'Local_Dendrite_Surface_IntDen': local_dend_surf_sum,
                'Area_Opt_Z_um2': area_opt_z_um2,
                'Geodesic_Distance_um': geo_dist_um, 
                'Vol_voxels': voxels, 
                'Vol_um3': vol,
                'Max_Intensity': max_intensity,
                'Sum_Intensity': sum_intensity,
                'Integrated_Density': int_density,
                'Z_Slices_Count': z_slices_count,
                'Avg_Initial_Dendrite_Intensity': state['avg_initial_dendrite_intensity'],
                'Barrier_um': current_barrier_val,
                'Tolerance': current_tolerance_val,
                'Z_Search_Range': current_zsearch_val,
                'Dendrite_Length_um': state['dendrite_length_um']
            })
            
    state['mask'] = combined_mask
    global final_results_df 
    final_results_df = pd.DataFrame(results_list)
    
    in_path = input_folder_input.value.strip()
    out_dir = os.path.join(in_path, 'output_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    current_file = state['files'][state['current_idx']]
    base_name = os.path.splitext(os.path.basename(current_file))[0]
    if base_name.endswith('.tif'):
        base_name = os.path.splitext(base_name)[0]
        
    csv_path = os.path.join(out_dir, base_name + '_spine_results.csv')
    filtered_img_path = os.path.join(out_dir, base_name + '_filtered.tif')
    mask_img_path = os.path.join(out_dir, base_name + '_segmentation_mask.tif')
    
    final_results_df.to_csv(csv_path, index=False)
    imwrite(filtered_img_path, current_smoothed_stack)
    imwrite(mask_img_path, combined_mask.astype(np.uint8) * 255)
    
    show_mask_cb.value = True
    refresh_display()
    
    mip_img_raw = np.max(state['raw_stack'], axis=0)
    mip_norm = mip_img_raw.astype(float)
    if mip_norm.max() > mip_norm.min():
        mip_norm = (255 * (mip_norm - mip_norm.min()) / (mip_norm.max() - mip_norm.min())).astype(np.uint8)
    else:
        mip_norm = np.zeros_like(mip_norm, dtype=np.uint8)
        
    mip_rgb = np.stack([mip_norm, mip_norm, mip_norm], axis=-1)
    
    mip_mask = np.max(combined_mask, axis=0)
    mip_rgb[mip_mask, 0] = 0   
    mip_rgb[mip_mask, 1] = 255 
    mip_rgb[mip_mask, 2] = 0   
    
    fig_mip, ax_mip = plt.subplots(figsize=(8, 8), dpi=150)
    ax_mip.imshow(mip_rgb)
    ax_mip.axis('off')
    
    ax_mip.set_ylim(mip_rgb.shape[0], 0)
    
    for _, row in final_results_df.iterrows():
        rx, ry = int(row['Corrected_X']), int(row['Corrected_Y'])
        tid = int(row['Target_ID'])
        
        c_class = row['Classification']
        if c_class == 'Filopodia':
            color = 'cyan'
        elif c_class == 'Suboptimal Measures':
            color = 'orange'
        else:
            color = 'yellow'
        
        ax_mip.plot(rx, ry, '.', color=color, label=str(tid))
        ax_mip.text(rx + 3, ry, str(tid), color=color, fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    mip_saved_path = os.path.join(out_dir, base_name + '_mip_segmented.png')
    fig_mip.savefig(mip_saved_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_mip)
    
    with log_output:
        print(f"✅ Batch Analysis Complete! Processed {len(final_results_df)} targets.")
        print(f"💾 Saved to folder: {os.path.basename(out_dir)}/")
        print(f"🖼️ Saved segmented MIP image: {base_name}_mip_segmented.png")
        display(HTML(final_results_df.to_html(index=False)))

def on_load_folder(b):
    in_path = input_folder_input.value.strip()
    if not os.path.isdir(in_path):
        with log_output: clear_output(); print("❌ Invalid input folder path.")
        return
        
    out_dir = os.path.join(in_path, 'output_analysis')
    os.makedirs(out_dir, exist_ok=True)
        
    all_files = sorted(glob(os.path.join(in_path, "*.tif")) + glob(os.path.join(in_path, "*.tiff")))
    if not all_files:
        with log_output: clear_output(); print("❌ No .tif files found in input folder.")
        return
        
    remaining_files = []
    for f in all_files:
        filename = os.path.basename(f).lower()
        if filename.endswith('_dendrite-geo.tif') or filename.endswith('_dendrite-respan.tif') or \
           filename.endswith('_dendrite-geo.tiff') or filename.endswith('_dendrite-respan.tiff'):
            continue
            
        base_name = os.path.splitext(os.path.basename(f))[0]
        if base_name.endswith('.tif'):
            base_name = os.path.splitext(base_name)[0]
        csv_check_path = os.path.join(out_dir, base_name + '_spine_results.csv')
        if not os.path.exists(csv_check_path):
            remaining_files.append(f)
            
    if not remaining_files:
        with log_output:
            clear_output()
            print("🎉 All images in this folder have already been analyzed in 'output_analysis'!")
        return
        
    state['files'] = remaining_files
    state['current_idx'] = 0
    load_image_at_index(0)

def on_next_image(b):
    if state['current_idx'] < len(state['files']) - 1:
        state['current_idx'] += 1
        load_image_at_index(state['current_idx'])
    else:
        with log_output:
            clear_output()
            print("🎉 Completed all remaining files in the queue!")

# Wire events
mask_source_dropdown.observe(on_mask_source_change, names='value')
wl_slider.observe(lambda c: img_display.set_clim(c['new'][0], c['new'][1]) or fig.canvas.draw_idle(), names='value')
barrier_slider.observe(on_barrier_change, names='value')
mode_radio.observe(lambda c: refresh_display(), names='value')

show_targets_cb.observe(lambda c: refresh_display(), names='value')
show_mask_cb.observe(lambda c: refresh_display(), names='value')
target_list_ui.observe(on_target_selected, names='value')

fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)
fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('scroll_event', on_scroll)

load_folder_btn.on_click(on_load_folder)
clear_paint_btn.on_click(on_clear_paint)
save_barrier_btn.on_click(on_save_barrier)
auto_seed_btn.on_click(auto_generate_seeds)
delete_target_btn.on_click(on_delete_selected_target)
apply_status_btn.on_click(on_apply_status)
rename_id_btn.on_click(on_rename_target)

respan_seed2_btn.on_click(lambda b: load_csv_seeds('seed2'))
respan_seed3_btn.on_click(lambda b: load_csv_seeds('seed3'))

save_target_btn.on_click(lambda b: on_save_target('spine'))
suboptimal_btn.on_click(lambda b: on_save_target('suboptimal'))
filopodia_btn.on_click(lambda b: on_save_target('filopodia'))

undo_target_btn.on_click(on_undo_target)
reset_view_btn.on_click(on_reset_view)
zoom_rect_btn.on_click(on_zoom_rect)
pan_btn.on_click(on_pan)
analyze_all_btn.on_click(on_analyze_all)
next_image_btn.on_click(on_next_image)

# ==========================================
# 4. TIGHT 3-COLUMN DASHBOARD LAYOUT
# ==========================================
col1 = widgets.VBox([
    input_folder_input, 
    widgets.HBox([load_folder_btn, file_info_label]),
    target_list_ui,
    widgets.HBox([status_dropdown, apply_status_btn]),
    delete_target_btn,
    widgets.HBox([rename_id_input, rename_id_btn]),
    seed_buttons_ui,
    widgets.HBox([save_target_btn, suboptimal_btn, filopodia_btn]),
    widgets.HBox([undo_target_btn, reset_view_btn]),
    widgets.HBox([zoom_rect_btn, pan_btn]),
    analyze_all_btn, next_image_btn,
    custom_id_input
], layout=widgets.Layout(width='380px', padding='0px 10px 0px 0px'))

col2 = widgets.VBox([
    fig.canvas
], layout=widgets.Layout(align_items='center'))

col3 = widgets.VBox([
    mask_source_dropdown,
    widgets.HBox([mode_radio, widgets.VBox([brush_size_slider, widgets.HBox([clear_paint_btn, save_barrier_btn])])]),
    z_slider, wl_slider, barrier_slider, tol_slider, z_search_slider, max_geodesic_slider,
    use_hessian_cb, strictness_slider, hessian_sigma_slider,
    widgets.HBox([show_targets_cb, show_mask_cb])
], layout=widgets.Layout(width='390px', padding='0px 0px 0px 10px'))

dashboard_ui = widgets.HBox([col1, col2, col3], layout=widgets.Layout(align_items='flex-start'))

display(widgets.VBox([dashboard_ui, log_output]))