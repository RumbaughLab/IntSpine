import os
import sys
import glob
import random
import traceback
import numpy as np
import pandas as pd
import h5py
import tifffile
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

from scipy import ndimage as ndi
from scipy.ndimage import convolve, uniform_filter
from skimage.filters import frangi, gaussian, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, binary_closing, disk, skeletonize, binary_dilation
from skimage.measure import regionprops, label
from skimage.segmentation import expand_labels

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QSlider, QCheckBox, QComboBox, QListWidget, QListWidgetItem,
                               QFileDialog, QGroupBox, QTableWidget, QTableWidgetItem, QAbstractItemView, QLineEdit)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. CORE PROCESSING FUNCTIONS
# ==========================================
def generate_hysteresis_mask(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        valid_pages = [page.asarray() for page in tif.pages]
        volume = np.stack(valid_pages)
        
    mip = np.max(volume, axis=0) if volume.ndim == 3 else volume
    mip_float = mip.astype(np.float64)
    mip_norm = (mip_float - np.min(mip_float)) / (np.max(mip_float) - np.min(mip_float) + 1e-8)

    mip_smoothed = gaussian(mip_norm, sigma=1.5)
    vesselness = frangi(mip_smoothed, sigmas=np.arange(1, 10, 2), black_ridges=False)

    high_thresh = np.percentile(vesselness, 98.0)
    low_thresh = np.percentile(vesselness, 94.0)
    
    hysteresis_mask = apply_hysteresis_threshold(vesselness, low_thresh, high_thresh)
    clean_binary_mask = remove_small_objects(hysteresis_mask, min_size=50)

    return mip, vesselness, clean_binary_mask

def polish_hybrid_mask(mip, mask1):
    closed_mask = binary_closing(mask1, disk(10))
    final_mask = remove_small_objects(closed_mask, min_size=500)
    return final_mask

def prune_skeleton_and_find_forks(binary_mask, prune_iterations=100):
    raw_skeleton = skeletonize(binary_mask)
    pruned_skeleton = raw_skeleton.copy()
    
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    
    for _ in range(prune_iterations):
        neighbor_count = convolve(pruned_skeleton.astype(int), kernel, mode='constant')
        endpoints = (neighbor_count == 11) & pruned_skeleton
        pruned_skeleton[endpoints] = False

    final_neighbor_count = convolve(pruned_skeleton.astype(int), kernel, mode='constant')
    fork_points_mask = (final_neighbor_count >= 13) & pruned_skeleton
    fork_coords = np.argwhere(fork_points_mask)
    
    return pruned_skeleton, fork_coords

def dilate_reskeletonize_and_find_nodes(pruned_skeleton):
    dilated_paths = binary_dilation(pruned_skeleton, disk(3))
    ultra_smooth_skeleton = skeletonize(dilated_paths)
    
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    neighbor_count = convolve(ultra_smooth_skeleton.astype(int), kernel, mode='constant')
    endpoints_coords = np.argwhere((neighbor_count == 11) & ultra_smooth_skeleton)
    
    raw_fork_mask = (neighbor_count >= 13) & ultra_smooth_skeleton
    labeled_forks, num_forks = ndi.label(raw_fork_mask)
    
    if num_forks > 0:
        centroids = ndi.center_of_mass(raw_fork_mask, labeled_forks, range(1, num_forks + 1))
        true_fork_coords = np.array(centroids)
    else:
        true_fork_coords = np.empty((0, 2))
    
    return ultra_smooth_skeleton, true_fork_coords, endpoints_coords

def extract_and_heal_graph(smooth_skeleton):
    clean_skeleton = smooth_skeleton.copy()
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    
    for _ in range(2):
        nc = convolve(clean_skeleton.astype(int), kernel, mode='constant')
        clean_skeleton[(nc == 11) & clean_skeleton] = False

    nc = convolve(clean_skeleton.astype(int), kernel, mode='constant')
    endpoints_coords = np.argwhere((nc == 11) & clean_skeleton)
    
    raw_fork_mask = (nc >= 13) & clean_skeleton
    labeled_forks, num_forks = ndi.label(raw_fork_mask)
    
    if num_forks > 0:
        centroids = ndi.center_of_mass(raw_fork_mask, labeled_forks, range(1, num_forks + 1))
        true_forks = np.array(centroids)
    else:
        true_forks = np.empty((0, 2))

    shattered_mask = clean_skeleton & ~raw_fork_mask
    labeled_segments, num_segments = ndi.label(shattered_mask, structure=np.ones((3,3)))
    
    healed_labels = expand_labels(labeled_segments, distance=3)
    final_labeled_skeleton = healed_labels * clean_skeleton

    if len(true_forks) > 0 and len(endpoints_coords) > 0:
        all_nodes_y = np.concatenate([true_forks[:, 0], endpoints_coords[:, 0]])
        all_nodes_x = np.concatenate([true_forks[:, 1], endpoints_coords[:, 1]])
    elif len(true_forks) > 0:
        all_nodes_y = true_forks[:, 0]
        all_nodes_x = true_forks[:, 1]
    elif len(endpoints_coords) > 0:
        all_nodes_y = endpoints_coords[:, 0]
        all_nodes_x = endpoints_coords[:, 1]
    else:
        all_nodes_y = np.array([])
        all_nodes_x = np.array([])
        
    unified_nodes = np.column_stack((all_nodes_y, all_nodes_x))
    
    return final_labeled_skeleton, num_segments, unified_nodes


# ==========================================
# 2. BATCH WORKER THREAD
# ==========================================
class BatchProcessorThread(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    
    def __init__(self, file_list, params):
        super().__init__()
        self.file_list = file_list
        self.params = params

    def run(self):
        all_results = []
        pad_xy = self.params['bbox_padding']
        
        for idx, filepath in enumerate(self.file_list):
            self.progress.emit(idx, f"Processing {os.path.basename(filepath)}...")
            try:
                mip, _, mask1 = generate_hysteresis_mask(filepath)
                final_polished_mask = polish_hybrid_mask(mip, mask1)
                clean_skeleton, _ = prune_skeleton_and_find_forks(final_polished_mask, prune_iterations=100)
                
                smooth_skeleton, true_forks, endpoints = dilate_reskeletonize_and_find_nodes(clean_skeleton)
                healed_labeled_skeleton, num_segments, all_nodes = extract_and_heal_graph(smooth_skeleton)

                with tifffile.TiffFile(filepath) as tif:
                    valid_pages = [page.asarray() for page in tif.pages]
                    stack = np.stack(valid_pages)
                if stack.ndim == 2: stack = np.expand_dims(stack, axis=0)

                total_smooth_length = int(np.sum(smooth_skeleton > 0))
                num_intersections = len(true_forks)
                
                soma_loc = self.params.get('soma_loc')
                if not soma_loc:
                    swin = self.params['soma_window']
                    lm = uniform_filter(mip.astype(float), size=swin)
                    sy, sx = np.unravel_index(np.argmax(lm), lm.shape)
                    soma_loc = (sx, sy)

                props = regionprops(healed_labeled_skeleton, intensity_image=mip)
                valid_props = []
                intensities = []
                
                active_ids = self.params.get('active_ids', [])
                
                for prop in props:
                    if prop.area < self.params['min_length']: continue
                    dist_soma = np.min(np.sqrt((prop.coords[:, 0] - soma_loc[0])**2 + (prop.coords[:, 1] - soma_loc[1])**2))
                    
                    if not active_ids: 
                        if self.params['dist_filter'] == 'Below Threshold' and dist_soma > self.params['max_soma_dist']: continue
                        if self.params['dist_filter'] == 'Above Threshold' and dist_soma <= self.params['max_soma_dist']: continue
                    else: 
                        if prop.label not in active_ids: continue

                    valid_props.append({"prop": prop, "soma_dist": dist_soma, "mean_int": prop.intensity_mean})
                    intensities.append(prop.intensity_mean)

                label_mapping = {}
                raw_labels = []
                if len(intensities) >= 3:
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    raw_labels = kmeans.fit_predict(np.array(intensities).reshape(-1, 1))
                    sorted_centers_idx = np.argsort(kmeans.cluster_centers_.flatten())
                    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_centers_idx)}

                cluster_names = {0: "Low", 1: "Medium", 2: "High"}

                out_dir = os.path.join(os.path.dirname(filepath), 'output_analysis')
                os.makedirs(out_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
                h5_path = os.path.join(out_dir, f"{base_name}_extracted.h5")
                
                with h5py.File(h5_path, 'w') as h5f:
                    h5f.attrs['soma_x'] = soma_loc[1]
                    h5f.attrs['soma_y'] = soma_loc[0]
                    h5f.attrs['total_smooth_skeleton_length_px'] = total_smooth_length
                    h5f.attrs['total_intersections'] = num_intersections
                    
                    h5f.create_dataset("mip", data=mip, compression="gzip")
                    h5f.create_dataset("smooth_skeleton", data=smooth_skeleton, compression="gzip")
                    h5f.create_dataset("healed_skeleton", data=healed_labeled_skeleton, compression="gzip")
                    h5f.create_dataset("true_forks", data=true_forks)
                    h5f.create_dataset("endpoints", data=endpoints)
                    
                    for i, data in enumerate(valid_props):
                        prop = data["prop"]
                        seg_id = prop.label
                        cluster_id = label_mapping.get(raw_labels[i], 1) if len(intensities) >= 3 else 1
                        int_class = cluster_names[cluster_id]
                        dist_soma = data['soma_dist']
                        
                        is_selected = False
                        if not active_ids:
                            is_selected = True
                            if self.params['dist_filter'] == 'Below Threshold' and dist_soma > self.params['max_soma_dist']: is_selected = False
                            if self.params['dist_filter'] == 'Above Threshold' and dist_soma <= self.params['max_soma_dist']: is_selected = False
                            if int_class == "Low" and not self.params.get('show_low', True): is_selected = False
                            if int_class == "Medium" and not self.params.get('show_med', True): is_selected = False
                            if int_class == "High" and not self.params.get('show_high', True): is_selected = False
                        else:
                            is_selected = seg_id in active_ids
                        
                        min_y, min_x, max_y, max_x = prop.bbox
                        t = int(max(0, min_y - pad_xy)); b = int(min(stack.shape[1], max_y + pad_xy))
                        l = int(max(0, min_x - pad_xy)); r = int(min(stack.shape[2], max_x + pad_xy))
                        sub_vol = stack[:, t:b, l:r]
                        
                        grp = h5f.create_group(f"segment_{seg_id}")
                        grp.create_dataset("volume", data=sub_vol, compression="gzip")
                        grp.attrs['bbox'] = [t, b, l, r]
                        grp.attrs['length_px'] = prop.area
                        grp.attrs['is_selected'] = is_selected
                        grp.attrs['cluster_id'] = cluster_id
                        grp.attrs['cluster_name'] = str(int_class)
                        grp.attrs['mean_intensity'] = data['mean_int']
                        grp.attrs['soma_dist'] = dist_soma
                        
                        if is_selected:
                            all_results.append({
                                'File': base_name, 'Seg_ID': seg_id,
                                'Length_px': prop.area, 'Intensity': int_class,
                                'Dist_to_Soma': round(dist_soma, 2),
                                'Total_Smooth_Skel_px': total_smooth_length, 'Nodes': num_intersections,
                                'Min_Len_Param': self.params['min_length'], 'Max_Dist_Param': self.params['max_soma_dist'],
                                'Dist_Filter': self.params['dist_filter']
                            })
            except Exception as e:
                self.progress.emit(idx, f"Error on {os.path.basename(filepath)}:\n{traceback.format_exc()}")
                
        df = pd.DataFrame(all_results)
        self.finished.emit(df)


# ==========================================
# 3. MAIN GUI CLASS
# ==========================================
class SpineExtractionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntSegment: Neurite Segmentation & Volumetric Extraction")
        self.resize(1600, 950)

        icon_path = os.path.join(os.path.dirname(__file__), 'intsegment_logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.input_folder = ''
        self.files = []
        self.current_idx = 0
        
        # Raw Data
        self.current_mip = None
        self.smooth_skeleton = None
        self.healed_skeleton = None
        self.raw_props = []
        self.test_forks = None
        self.test_endpoints = None
        self.soma_loc = None
        self.total_smooth_length = 0
        
        # Filtered Data (Updates dynamically)
        self.segment_data = [] 
        
        # Track matplotlib objects
        self.img_display = None
        self.seg_overlay = None 
        self.skel_overlay = None
        self.node_scatter = None
        self.end_scatter = None
        self.soma_scatter = None
        self.dist_circle = None
        
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- LEFT COLUMN (Routing / Data Input) ---
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        folder_group = QGroupBox("Data Input")
        folder_layout = QVBoxLayout()
        
        btn_layout1 = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Folder")
        self.btn_load = QPushButton("Load Remaining")
        btn_layout1.addWidget(self.btn_browse)
        btn_layout1.addWidget(self.btn_load)
        folder_layout.addLayout(btn_layout1)
        
        btn_layout2 = QHBoxLayout()
        self.btn_load_analyzed = QPushButton("Load Analyzed")
        self.btn_batch = QPushButton("Batch Unattended")
        self.btn_load_analyzed.setStyleSheet("background-color: #E65100; color: white;")
        self.btn_batch.setStyleSheet("background-color: #B71C1C; color: white;")
        btn_layout2.addWidget(self.btn_load_analyzed)
        btn_layout2.addWidget(self.btn_batch)
        folder_layout.addLayout(btn_layout2)
        
        self.lbl_file_info = QLabel("No folder loaded")
        folder_layout.addWidget(self.lbl_file_info)
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)
        
        self.btn_analyze_current = QPushButton("Analyze & Save Targets")
        self.btn_analyze_current.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        left_layout.addWidget(self.btn_analyze_current)
        
        nav_img_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev Image")
        self.btn_prev.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        self.btn_next = QPushButton("Next Image >")
        self.btn_next.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        nav_img_layout.addWidget(self.btn_prev)
        nav_img_layout.addWidget(self.btn_next)
        left_layout.addLayout(nav_img_layout)

        # Random Selection Box
        rand_lyt = QHBoxLayout()
        self.txt_rand_n = QLineEdit()
        self.txt_rand_n.setPlaceholderText("N segments")
        self.btn_rand = QPushButton("Select Random N")
        rand_lyt.addWidget(self.txt_rand_n)
        rand_lyt.addWidget(self.btn_rand)
        left_layout.addLayout(rand_lyt)

        left_layout.addWidget(QLabel("Target List (Ctrl+Click to multiselect):"))
        self.list_targets = QListWidget()
        self.list_targets.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(self.list_targets)

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(["ID", "Len(px)", "Dist to Soma", "Intensity"])
        self.table_results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        left_layout.addWidget(QLabel("Live Preview Table (Active Selection):"))
        left_layout.addWidget(self.table_results)
        
        # --- CENTER COLUMN (Matplotlib Canvas) ---
        center_layout = QVBoxLayout()
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.subplots_adjust(bottom=0.02, top=0.98, left=0.02, right=0.98)
        
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.ax.axis('off')
        
        center_layout.addWidget(self.toolbar)
        center_layout.addWidget(self.canvas)
        
        # --- RIGHT COLUMN (Controls) ---
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        settings_group = QGroupBox("Dynamic Processing Filters")
        set_lyt = QVBoxLayout()
        
        self.lyt_min_len, self.sld_min_len = self.create_slider("Min Seg Length", 10, 500, 50)
        self.lyt_pad, self.sld_pad = self.create_slider("BBox Padding (px)", 5, 100, 30)
        self.lyt_soma, self.sld_soma = self.create_slider("Soma Window", 5, 100, 20)
        self.lyt_max_dist, self.sld_max_dist = self.create_slider("Max Distance Threshold", 10, 2000, 800)
        
        self.combo_dist_filter = QComboBox()
        self.combo_dist_filter.addItems(["All", "Below Threshold", "Above Threshold"])
        self.combo_dist_filter.setCurrentText("Below Threshold")
        set_lyt.addWidget(QLabel("Distance Filter:"))
        set_lyt.addWidget(self.combo_dist_filter)

        int_group = QGroupBox("Intensity Clusters")
        int_lyt = QHBoxLayout()
        self.chk_int_low = QCheckBox("Low")
        self.chk_int_med = QCheckBox("Medium")
        self.chk_int_high = QCheckBox("High")
        self.chk_int_low.setChecked(True); self.chk_int_med.setChecked(True); self.chk_int_high.setChecked(True)
        int_lyt.addWidget(self.chk_int_low); int_lyt.addWidget(self.chk_int_med); int_lyt.addWidget(self.chk_int_high)
        int_group.setLayout(int_lyt)
        set_lyt.addWidget(int_group)
        
        set_lyt.addLayout(self.lyt_min_len)
        set_lyt.addLayout(self.lyt_pad)
        set_lyt.addLayout(self.lyt_soma)
        set_lyt.addLayout(self.lyt_max_dist)
        settings_group.setLayout(set_lyt)
        right_layout.addWidget(settings_group)
        
        contrast_group = QGroupBox("Contrast Settings")
        contrast_lyt = QVBoxLayout()
        self.lyt_wl_min, self.sld_wl_min = self.create_slider("Contrast Min", 0, 65535, 0)
        self.lyt_wl_max, self.sld_wl_max = self.create_slider("Contrast Max", 0, 65535, 65535)
        self.btn_auto_contrast = QPushButton("Auto Contrast")
        self.btn_auto_contrast.setStyleSheet("background-color: #555; color: white;")
        contrast_lyt.addLayout(self.lyt_wl_min)
        contrast_lyt.addLayout(self.lyt_wl_max)
        contrast_lyt.addWidget(self.btn_auto_contrast)
        contrast_group.setLayout(contrast_lyt)
        right_layout.addWidget(contrast_group)
        
        disp_group = QGroupBox("Display Overlays")
        d_layout = QVBoxLayout()
        self.chk_skel = QCheckBox("Show Smoothed Full Skeleton")
        self.chk_segs = QCheckBox("Show Segments (Clustered)")
        self.chk_nodes = QCheckBox("Show Branch Points/Nodes")
        self.chk_bbox = QCheckBox("Show Bounding Boxes")
        self.chk_origin = QCheckBox("Show Origin & Threshold")
        
        self.chk_skel.setChecked(False); self.chk_segs.setChecked(True)
        self.chk_nodes.setChecked(True); self.chk_bbox.setChecked(False)
        self.chk_origin.setChecked(True)
        
        for chk in [self.chk_skel, self.chk_segs, self.chk_nodes, self.chk_bbox, self.chk_origin]:
            d_layout.addWidget(chk)
            
        disp_group.setLayout(d_layout)
        right_layout.addWidget(disp_group)

        self.btn_test = QPushButton("Test Parameters on Image")
        self.btn_test.setStyleSheet("background-color: #005A99; color: white; font-weight: bold;")
        right_layout.addWidget(self.btn_test)

        self.lbl_feedback = QLabel("Status: Ready")
        self.lbl_feedback.setWordWrap(True)
        right_layout.addWidget(self.lbl_feedback)
        right_layout.addStretch()
        
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(center_layout, 6)
        main_layout.addLayout(right_layout, 2)

        self.connect_signals()

    def create_slider(self, label, min_val, max_val, default):
        lyt = QHBoxLayout()
        lbl_name = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        lbl_val = QLabel(str(default))
        slider.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))
        lyt.addWidget(lbl_name)
        lyt.addWidget(slider)
        lyt.addWidget(lbl_val)
        return lyt, slider

    def apply_dark_theme(self):
        dark_stylesheet = """
        QMainWindow { background-color: #121212; color: #ffffff; }
        QWidget { background-color: #121212; color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
        QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #aaa; }
        QPushButton { background-color: #333; border: 1px solid #555; padding: 6px; border-radius: 4px; color: white; }
        QPushButton:hover { background-color: #444; border: 1px solid #888; }
        QLineEdit, QTableWidget, QListWidget, QComboBox { background-color: #1e1e1e; border: 1px solid #555; color: white; outline: none; }
        QListWidget::item:selected { background-color: #007ACC; color: white; }
        QHeaderView::section { background-color: #333; padding: 4px; border: 1px solid #555; font-weight: bold; }
        QSlider::groove:horizontal { border: 1px solid #3A3939; height: 8px; background: #201F1F; border-radius: 4px; }
        QSlider::handle:horizontal { background: #007ACC; border: 1px solid #005A99; width: 14px; margin: -3px 0; border-radius: 7px; }
        """
        self.setStyleSheet(dark_stylesheet)

    def connect_signals(self):
        self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_load.clicked.connect(self.on_load_remaining)
        self.btn_load_analyzed.clicked.connect(self.on_load_analyzed)
        self.btn_batch.clicked.connect(self.on_batch_unattended)
        self.btn_analyze_current.clicked.connect(self.on_analyze_current)
        self.btn_prev.clicked.connect(self.on_prev_image)
        self.btn_next.clicked.connect(self.on_next_image)
        self.btn_test.clicked.connect(self.run_test)
        self.btn_rand.clicked.connect(self.on_select_random_n)
        self.btn_auto_contrast.clicked.connect(self.on_auto_contrast)
        
        self.sld_wl_min.sliderReleased.connect(self.update_contrast)
        self.sld_wl_max.sliderReleased.connect(self.update_contrast)
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # Dynamic Filtering Signals
        self.sld_min_len.sliderReleased.connect(self.apply_filters_and_clustering)
        self.sld_max_dist.sliderReleased.connect(self.apply_filters_and_clustering)
        self.combo_dist_filter.currentTextChanged.connect(self.apply_filters_and_clustering)
        self.chk_int_low.toggled.connect(self.apply_filters_and_clustering)
        self.chk_int_med.toggled.connect(self.apply_filters_and_clustering)
        self.chk_int_high.toggled.connect(self.apply_filters_and_clustering)

        # Display Checkbox Signals
        self.chk_skel.toggled.connect(self.draw_overlays)
        self.chk_segs.toggled.connect(self.draw_overlays)
        self.chk_nodes.toggled.connect(self.draw_overlays)
        self.chk_bbox.toggled.connect(self.draw_overlays)
        self.chk_origin.toggled.connect(self.draw_overlays)
        self.sld_pad.sliderReleased.connect(self.draw_overlays)

        self.list_targets.itemSelectionChanged.connect(self.update_preview_table)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if folder:
            self.input_folder = folder
            self.files = sorted(glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff")))
            if self.files:
                self.current_idx = 0
                self.load_image(self.files[self.current_idx])
                self.lbl_file_info.setText(f"Folder: {os.path.basename(folder)}\n{len(self.files)} files found.")

    def on_load_remaining(self):
        if not self.input_folder: return
        out_dir = os.path.join(self.input_folder, 'output_analysis')
        all_f = sorted(glob.glob(os.path.join(self.input_folder, "*.tif")) + glob.glob(os.path.join(self.input_folder, "*.tiff")))
        self.files = [f for f in all_f if not os.path.exists(os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + "_extracted.h5"))]
        if self.files:
            self.current_idx = 0
            self.load_image(self.files[0])
            self.lbl_file_info.setText(f"Loaded {len(self.files)} remaining files.")

    def on_load_analyzed(self):
        if not self.input_folder: return
        out_dir = os.path.join(self.input_folder, 'output_analysis')
        all_f = sorted(glob.glob(os.path.join(self.input_folder, "*.tif")) + glob.glob(os.path.join(self.input_folder, "*.tiff")))
        self.files = [f for f in all_f if os.path.exists(os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + "_extracted.h5"))]
        if self.files:
            self.current_idx = 0
            self.load_image(self.files[0])
            self.lbl_file_info.setText(f"Loaded {len(self.files)} analyzed files.")

    def update_contrast(self):
        if self.current_mip is not None and getattr(self, 'img_display', None) is not None:
            vmin = self.sld_wl_min.value()
            vmax = self.sld_wl_max.value()
            if vmin < vmax: 
                self.img_display.set_clim(vmin, vmax)
                self.fig.canvas.draw_idle()

    def on_auto_contrast(self):
        if self.current_mip is not None:
            p_low = np.percentile(self.current_mip, 2)
            p_high = np.percentile(self.current_mip, 99.5)
            vmax_pop = p_high * 1.2
            if p_low >= vmax_pop: vmax_pop = p_low + 1

            self.sld_wl_min.blockSignals(True)
            self.sld_wl_max.blockSignals(True)
            self.sld_wl_min.setValue(int(p_low))
            self.sld_wl_max.setValue(int(vmax_pop))
            self.sld_wl_min.blockSignals(False)
            self.sld_wl_max.blockSignals(False)

            self.update_contrast()

    def load_image(self, filepath):
        self.lbl_feedback.setText(f"Loading {os.path.basename(filepath)}...")
        QApplication.processEvents()
        
        with tifffile.TiffFile(filepath) as tif:
            valid_pages = [page.asarray() for page in tif.pages]
            vol = np.stack(valid_pages)
        self.current_mip = np.max(vol, axis=0) if vol.ndim == 3 else vol

        # Auto-scale initial contrast
        p_low = np.percentile(self.current_mip, 2)
        p_high = np.percentile(self.current_mip, 99.5)
        vmax_pop = p_high * 1.2
        if p_low >= vmax_pop: vmax_pop = p_low + 1

        self.sld_wl_min.blockSignals(True)
        self.sld_wl_max.blockSignals(True)
        
        slider_max = int(max(self.current_mip.max(), vmax_pop * 1.5))
        self.sld_wl_min.setRange(int(self.current_mip.min()), slider_max)
        self.sld_wl_max.setRange(int(self.current_mip.min()), slider_max)
        self.sld_wl_min.setValue(int(p_low))
        self.sld_wl_max.setValue(int(vmax_pop))
        
        self.sld_wl_min.blockSignals(False)
        self.sld_wl_max.blockSignals(False)

        self.smooth_skeleton = None
        self.healed_skeleton = None
        self.raw_props = []
        self.segment_data = []
        self.test_forks = None
        self.test_endpoints = None
        self.soma_loc = None
        
        self.list_targets.blockSignals(True)
        self.list_targets.clear()
        self.list_targets.blockSignals(False)
        self.table_results.setRowCount(0)
        
        # Completely redraw axis to enforce contrast scaling safely
        self.ax.clear()
        self.img_display = self.ax.imshow(self.current_mip, cmap='gray', vmin=p_low, vmax=vmax_pop)
        self.ax.axis('off')

        out_dir = os.path.join(os.path.dirname(filepath), 'output_analysis')
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        h5_path = os.path.join(out_dir, f"{base_name}_extracted.h5")

        if os.path.exists(h5_path):
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    self.soma_loc = (h5f.attrs['soma_y'], h5f.attrs['soma_x'])
                    self.smooth_skeleton = h5f['smooth_skeleton'][:]
                    self.healed_skeleton = h5f['healed_skeleton'][:]
                    self.test_forks = h5f['true_forks'][:]
                    self.test_endpoints = h5f['endpoints'][:]
                    
                    self.raw_props = regionprops(self.healed_skeleton, intensity_image=self.current_mip)
                    props_by_label = {p.label: p for p in self.raw_props}
                    
                    active_ids = []
                    self.segment_data = []
                    
                    for key in h5f.keys():
                        if key.startswith('segment_'):
                            grp = h5f[key]
                            seg_id = int(key.split('_')[1])
                            prop = props_by_label.get(seg_id)
                            if prop is None: continue
                            
                            c_id = grp.attrs['cluster_id']
                            c_name = str(grp.attrs['cluster_name'])
                            is_selected = grp.attrs.get('is_selected', False)
                            
                            data_dict = {
                                "prop": prop,
                                "seg_id": seg_id,
                                "soma_dist": grp.attrs['soma_dist'],
                                "mean_intensity": grp.attrs['mean_intensity'],
                                "cluster_id": c_id,
                                "cluster_name": c_name
                            }
                            self.segment_data.append(data_dict)
                            if is_selected: active_ids.append(seg_id)
                    
                    self.list_targets.blockSignals(True)
                    for data in self.segment_data:
                        item_text = f"ID: {data['seg_id']} | Len: {data['prop'].area} | {data['cluster_name']} Int | Dist: {int(data['soma_dist'])}"
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.UserRole, data)
                        self.list_targets.addItem(item)
                        if data['seg_id'] in active_ids:
                            item.setSelected(True)
                    self.list_targets.blockSignals(False)
                    
                    self.update_preview_table()
                    self.draw_overlays()
                    self.lbl_feedback.setText(f"Loaded analyzed data from HDF5.")
            except Exception as e:
                self.lbl_feedback.setText(f"Failed to load HDF5: {str(e)}\n{traceback.format_exc()}")
                self.draw_overlays()
        else:
            self.draw_overlays()
            self.lbl_feedback.setText(f"Loaded {os.path.basename(filepath)}. Hit Test Parameters to extract graph.")

    def on_canvas_click(self, event):
        if event.button == 1 and event.inaxes == self.ax and self.current_mip is not None:
            self.soma_loc = (int(event.ydata), int(event.xdata))
            if self.raw_props:
                self.apply_filters_and_clustering()
            else:
                self.draw_overlays()
            self.lbl_feedback.setText(f"Manual Origin set to X:{self.soma_loc[1]}, Y:{self.soma_loc[0]}")

    def run_test(self):
        if self.current_mip is None: return
        self.lbl_feedback.setText("Testing Parameters: Extracting Graph (Heavy Process)...")
        QApplication.processEvents()
        
        if self.soma_loc is None:
            swin = self.sld_soma.value()
            lm = uniform_filter(self.current_mip.astype(float), size=swin)
            sy, sx = np.unravel_index(np.argmax(lm), lm.shape)
            self.soma_loc = (sy, sx)

        try:
            filepath = self.files[self.current_idx]
            mip, _, mask1 = generate_hysteresis_mask(filepath)
            final_polished_mask = polish_hybrid_mask(mip, mask1)
            
            clean_skeleton, _ = prune_skeleton_and_find_forks(final_polished_mask, prune_iterations=100)
            self.smooth_skeleton, self.test_forks, self.test_endpoints = dilate_reskeletonize_and_find_nodes(clean_skeleton)
            self.total_smooth_length = int(np.sum(self.smooth_skeleton > 0))
            self.healed_skeleton, _, _ = extract_and_heal_graph(self.smooth_skeleton)
            
            self.raw_props = regionprops(self.healed_skeleton, intensity_image=self.current_mip)
            
            self.apply_filters_and_clustering()
            self.lbl_feedback.setText("Graph Extraction Complete. You can now adjust filters instantly.")
        except Exception as e:
            self.lbl_feedback.setText(f"Error during testing: {str(e)}\n{traceback.format_exc()}")

    def apply_filters_and_clustering(self):
        if not self.raw_props or self.soma_loc is None: return
        
        self.segment_data = []
        intensities = []
        dist_mode = self.combo_dist_filter.currentText()
        max_dist = self.sld_max_dist.value()
        min_len = self.sld_min_len.value()
        
        for prop in self.raw_props:
            if prop.area < min_len: continue
            dist_soma = np.min(np.sqrt((prop.coords[:, 0] - self.soma_loc[0])**2 + (prop.coords[:, 1] - self.soma_loc[1])**2))
            
            if dist_mode == 'Below Threshold' and dist_soma > max_dist: continue
            if dist_mode == 'Above Threshold' and dist_soma <= max_dist: continue
            
            self.segment_data.append({
                "prop": prop,
                "seg_id": prop.label,
                "soma_dist": dist_soma,
                "mean_intensity": prop.intensity_mean
            })
            intensities.append(prop.intensity_mean)

        label_mapping = {}
        raw_labels = []
        if len(intensities) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            raw_labels = kmeans.fit_predict(np.array(intensities).reshape(-1, 1))
            sorted_centers_idx = np.argsort(kmeans.cluster_centers_.flatten())
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_centers_idx)}
        else:
            raw_labels = [1] * len(intensities)
            label_mapping = {1: 1}
            
        cluster_names = {0: "Low", 1: "Medium", 2: "High"}
        
        self.list_targets.blockSignals(True)
        self.list_targets.clear()
        
        show_low = self.chk_int_low.isChecked()
        show_med = self.chk_int_med.isChecked()
        show_high = self.chk_int_high.isChecked()

        for i, data in enumerate(self.segment_data):
            cid = label_mapping.get(raw_labels[i], 1)
            int_class = cluster_names[cid]
            
            if int_class == "Low" and not show_low: continue
            if int_class == "Medium" and not show_med: continue
            if int_class == "High" and not show_high: continue
            
            data["cluster_id"] = cid
            data["cluster_name"] = int_class
            
            item_text = f"ID: {data['seg_id']} | Len: {data['prop'].area} | {int_class} Int | Dist: {int(data['soma_dist'])}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, data)
            self.list_targets.addItem(item)
            
        self.list_targets.selectAll()
        self.list_targets.blockSignals(False)
        self.update_preview_table()
        self.draw_overlays()

    def on_select_random_n(self):
        val = self.txt_rand_n.text().strip()
        if not val.isdigit(): return
        n = int(val)
        
        count = self.list_targets.count()
        if count == 0: return
        
        n = min(n, count)
        indices = random.sample(range(count), n)
        
        self.list_targets.blockSignals(True)
        self.list_targets.clearSelection()
        for i in indices:
            self.list_targets.item(i).setSelected(True)
        self.list_targets.blockSignals(False)
        self.update_preview_table()

    def update_preview_table(self):
        active_items = self.list_targets.selectedItems()
        self.table_results.setRowCount(len(active_items))
        
        for row_idx, item in enumerate(active_items):
            data = item.data(Qt.UserRole)
            self.table_results.setItem(row_idx, 0, QTableWidgetItem(str(data['seg_id'])))
            self.table_results.setItem(row_idx, 1, QTableWidgetItem(str(data['prop'].area)))
            self.table_results.setItem(row_idx, 2, QTableWidgetItem(str(int(data['soma_dist']))))
            self.table_results.setItem(row_idx, 3, QTableWidgetItem(str(data['cluster_name'])))
        self.draw_overlays()

    def draw_overlays(self):
        [p.remove() for p in reversed(self.ax.patches)]
        [t.remove() for t in reversed(self.ax.texts)]
        
        if hasattr(self, 'seg_overlay') and self.seg_overlay is not None:
            try: self.seg_overlay.remove(); self.seg_overlay = None
            except: pass
        if hasattr(self, 'skel_overlay') and self.skel_overlay is not None:
            try: self.skel_overlay.remove(); self.skel_overlay = None
            except: pass
        if hasattr(self, 'node_scatter') and self.node_scatter is not None:
            try: self.node_scatter.remove(); self.node_scatter = None
            except: pass
        if hasattr(self, 'end_scatter') and self.end_scatter is not None:
            try: self.end_scatter.remove(); self.end_scatter = None
            except: pass
        if hasattr(self, 'soma_scatter') and self.soma_scatter is not None:
            try: self.soma_scatter.remove(); self.soma_scatter = None
            except: pass
        if hasattr(self, 'dist_circle') and self.dist_circle is not None:
            try: self.dist_circle.remove(); self.dist_circle = None
            except: pass

        legend_elements = []

        if self.chk_skel.isChecked() and self.smooth_skeleton is not None:
            vis_skel = expand_labels(self.smooth_skeleton, distance=1)
            masked_skel = np.ma.masked_where(vis_skel == 0, vis_skel)
            skel_cmap = ListedColormap(['#FF00FF']) # Magenta for high visibility
            self.skel_overlay = self.ax.imshow(masked_skel, cmap=skel_cmap, alpha=0.8, interpolation='nearest')
            legend_elements.append(patches.Patch(color='#FF00FF', label='Smoothed Skeleton'))

        if self.healed_skeleton is not None:
            active_ids = [item.data(Qt.UserRole)['seg_id'] for item in self.list_targets.selectedItems()]
            
            if self.chk_segs.isChecked() and active_ids:
                visual_cluster_mask = np.zeros_like(self.healed_skeleton)
                visual_segments = expand_labels(self.healed_skeleton, distance=2)
                
                for i in range(self.list_targets.count()):
                    data = self.list_targets.item(i).data(Qt.UserRole)
                    if data['seg_id'] in active_ids:
                        visual_cluster_mask[visual_segments == data['seg_id']] = data['cluster_id'] + 1
                        
                custom_colors = [
                    [0, 0, 0, 0],         
                    [0, 1, 1, 1.0],       # Low = Cyan
                    [1, 1, 0, 1.0],       # Medium = Yellow
                    [1, 0.2, 0.2, 1.0]    # High = Red
                ]
                intensity_cmap = ListedColormap(custom_colors)
                masked_visuals = np.ma.masked_where(visual_cluster_mask == 0, visual_cluster_mask)
                self.seg_overlay = self.ax.imshow(masked_visuals, cmap=intensity_cmap, interpolation='nearest', vmin=0, vmax=3)
                
                if self.chk_int_low.isChecked(): legend_elements.append(patches.Patch(color=[0, 1, 1], label='Low Intensity'))
                if self.chk_int_med.isChecked(): legend_elements.append(patches.Patch(color=[1, 1, 0], label='Medium Intensity'))
                if self.chk_int_high.isChecked(): legend_elements.append(patches.Patch(color=[1, 0.2, 0.2], label='High Intensity'))

            pad = self.sld_pad.value()
            for i in range(self.list_targets.count()):
                data = self.list_targets.item(i).data(Qt.UserRole)
                if data['seg_id'] in active_ids:
                    prop = data['prop']
                    if self.chk_segs.isChecked():
                        geom_center = prop.centroid
                        distances_to_center = np.sum((prop.coords - geom_center)**2, axis=1)
                        mid_y, mid_x = prop.coords[np.argmin(distances_to_center)]
                        
                        self.ax.text(mid_x, mid_y, str(prop.label), color='white', fontsize=10, 
                                     fontweight='bold', ha='center', va='center', 
                                     bbox=dict(facecolor='black', alpha=0.8, edgecolor='white', pad=1.5, linewidth=0.8))
                    
                    if self.chk_bbox.isChecked():
                        min_y, min_x, max_y, max_x = prop.bbox
                        t = int(max(0, min_y - pad)); b = int(min(self.current_mip.shape[0], max_y + pad))
                        l = int(max(0, min_x - pad)); r = int(min(self.current_mip.shape[1], max_x + pad))
                        rect = patches.Rectangle((l, t), r-l, b-t, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
                        self.ax.add_patch(rect)

        if self.chk_nodes.isChecked():
            if self.test_forks is not None and len(self.test_forks) > 0:
                self.node_scatter = self.ax.scatter(self.test_forks[:, 1], self.test_forks[:, 0], color='cyan', s=45, zorder=5)
                legend_elements.insert(0, Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Forks'))
            
            if self.test_endpoints is not None and len(self.test_endpoints) > 0:
                self.end_scatter = self.ax.scatter(self.test_endpoints[:, 1], self.test_endpoints[:, 0], color='red', s=45, zorder=6)
                legend_elements.insert(0, Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Endpoints'))
                
        if self.chk_origin.isChecked() and self.soma_loc is not None:
            self.soma_scatter = self.ax.scatter(self.soma_loc[1], self.soma_loc[0], color='white', marker='*', s=250, edgecolors='black', zorder=10)
            legend_elements.insert(0, Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='k', markersize=12, label='Soma Origin'))
            
            self.dist_circle = patches.Circle((self.soma_loc[1], self.soma_loc[0]), self.sld_max_dist.value(), 
                                         color='white', fill=False, linestyle='--', linewidth=2, alpha=0.7)
            self.ax.add_patch(self.dist_circle)
            legend_elements.insert(1, Line2D([0], [0], color='w', linestyle='--', linewidth=2, label='Max Distance Threshold'))
            
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)
        else:
            if self.ax.get_legend(): self.ax.get_legend().remove()

        # Enforce strict image boundaries so threshold circles don't zoom out the view
        if self.current_mip is not None:
            self.ax.set_xlim(0, self.current_mip.shape[1])
            self.ax.set_ylim(self.current_mip.shape[0], 0)

        self.fig.canvas.draw_idle()

    def on_analyze_current(self):
        if not self.files: return
        active_ids = [item.data(Qt.UserRole)['seg_id'] for item in self.list_targets.selectedItems()]
        if not active_ids:
            self.lbl_feedback.setText("No segments selected in the target list!")
            return
        self.lbl_feedback.setText(f"Analyzing {os.path.basename(self.files[self.current_idx])}...")
        self.run_batch_logic([self.files[self.current_idx]])

    def on_batch_unattended(self):
        if not self.files: return
        self.lbl_feedback.setText(f"Launching batch analysis for {len(self.files)} files...")
        self.run_batch_logic(self.files)

    def run_batch_logic(self, file_list):
        self.btn_batch.setEnabled(False)
        self.btn_analyze_current.setEnabled(False)
        
        active_ids = []
        if len(file_list) == 1:
            active_ids = [item.data(Qt.UserRole)['seg_id'] for item in self.list_targets.selectedItems()]

        params = {
            'min_length': self.sld_min_len.value(),
            'bbox_padding': self.sld_pad.value(),
            'soma_window': self.sld_soma.value(),
            'max_soma_dist': self.sld_max_dist.value(),
            'dist_filter': self.combo_dist_filter.currentText(),
            'show_low': self.chk_int_low.isChecked(),
            'show_med': self.chk_int_med.isChecked(),
            'show_high': self.chk_int_high.isChecked(),
            'soma_loc': self.soma_loc if len(file_list) == 1 else None,
            'active_ids': active_ids
        }
        
        self.worker = BatchProcessorThread(file_list, params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_batch_finished)
        self.worker.start()

    def update_progress(self, idx, msg):
        self.lbl_feedback.setText(f"[{idx+1}] {msg}")

    def on_batch_finished(self, df):
        self.btn_batch.setEnabled(True)
        self.btn_analyze_current.setEnabled(True)
        self.lbl_feedback.setText("Processing complete! HDF5 files and data table saved.")
        
        self.table_results.setColumnCount(5)
        self.table_results.setHorizontalHeaderLabels(["File", "Seg_ID", "Len(px)", "Intensity", "Params"])
        self.table_results.setRowCount(0) 
        self.table_results.setRowCount(len(df))
        for row_idx, row in df.iterrows():
            params_str = f"MinL:{row.get('Min_Len_Param','N/A')} D:{row.get('Max_Dist_Param','N/A')}"
            self.table_results.setItem(row_idx, 0, QTableWidgetItem(str(row['File'])))
            self.table_results.setItem(row_idx, 1, QTableWidgetItem(str(row['Seg_ID'])))
            self.table_results.setItem(row_idx, 2, QTableWidgetItem(str(row['Length_px'])))
            self.table_results.setItem(row_idx, 3, QTableWidgetItem(str(row['Intensity'])))
            self.table_results.setItem(row_idx, 4, QTableWidgetItem(params_str))
            
        out_csv = os.path.join(self.input_folder, 'output_analysis', 'global_results.csv')
        df.to_csv(out_csv, index=False)

    def on_prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image(self.files[self.current_idx])

    def on_next_image(self):
        if self.current_idx < len(self.files) - 1:
            self.current_idx += 1
            self.load_image(self.files[self.current_idx])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(__file__), 'intsegment_logo.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = SpineExtractionApp()
    window.show()
    sys.exit(app.exec())