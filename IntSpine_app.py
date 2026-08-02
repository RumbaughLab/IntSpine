import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from tifffile import imread, imwrite
from scipy.ndimage import distance_transform_edt, label, binary_fill_holes, uniform_filter, maximum_position, binary_erosion, gaussian_filter, binary_dilation, convolve
from scipy.spatial.distance import pdist, squareform
from skimage.draw import disk
from skimage import measure, morphology, filters, graph
from skimage.filters import gaussian  

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QLineEdit, QComboBox, QSlider, QCheckBox, 
                               QListWidget, QAbstractItemView, QTextEdit, QFileDialog, QRadioButton, QButtonGroup, QGroupBox)
from PySide6.QtGui import QShortcut, QKeySequence, QIcon
from PySide6.QtCore import Qt
import ctypes

# ==========================================
# 1. CORE APPLICATION CLASS
# ==========================================
class SpineAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntSpine: Volumetric Interactive Spine Analyzer")
        self.resize(1600, 900)
        
        icon_path = os.path.join(os.path.dirname(__file__), 'intspine_logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # State Variables
        self.dz, self.dy, self.dx = 0.3, 0.108, 0.108  
        self.voxel_volume = self.dx * self.dy * self.dz
        
        self.input_folder = ''
        self.files = []
        self.current_idx = 0
        
        self.raw_stack = None
        self.base_smoothed_stack = None
        self.dist_field_3d = None
        self.dendrite_length_um = 0.0
        self.initial_shaft_barrier = None
        self.avg_initial_dendrite_intensity = 0.0
        
        self.z = 0
        self.click_x = None; self.click_y = None       
        self.target_x = None; self.target_y = None; self.target_z = None 
        self.mask = None 
        self.shaft_barrier = None
        
        self.painted_barrier_2d = None
        self.erased_barrier_2d = None
        self.is_drawing = False
        
        self.saved_targets = []
        self.target_counter = 1
        self.is_correction_mode = False
        self.loaded_df = None
        
        self.texts_ax1, self.texts_ax2 = [], []
        self.dots_ax1, self.dots_ax2 = [], []
        
        self.init_ui()
        self.apply_dark_theme()

    # ==========================================
    # 2. UI SETUP
    # ==========================================
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- LEFT COLUMN (Data & Targets) ---
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
        
        target_group = QGroupBox("Target List")
        target_layout = QVBoxLayout()
        self.list_targets = QListWidget()
        self.list_targets.setSelectionMode(QAbstractItemView.ExtendedSelection)
        target_layout.addWidget(self.list_targets)
        
        status_layout = QHBoxLayout()
        self.combo_status = QComboBox()
        self.combo_status.addItems(['static', 'new', 'eliminated'])
        self.btn_apply_status = QPushButton("Apply Status")
        status_layout.addWidget(self.combo_status)
        status_layout.addWidget(self.btn_apply_status)
        target_layout.addLayout(status_layout)
        
        self.btn_delete_target = QPushButton("Delete Selected Targets")
        target_layout.addWidget(self.btn_delete_target)
        
        rename_layout = QHBoxLayout()
        self.input_rename = QLineEdit()
        self.input_rename.setPlaceholderText("New ID")
        self.btn_rename = QPushButton("Update Target ID")
        rename_layout.addWidget(self.input_rename)
        rename_layout.addWidget(self.btn_rename)
        target_layout.addLayout(rename_layout)
        target_group.setLayout(target_layout)
        left_layout.addWidget(target_group)
        
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        seed_layout = QHBoxLayout()
        self.btn_auto_seed = QPushButton("Auto-Seed")
        self.btn_csv_seed2 = QPushButton("CSV Seed 2")
        self.btn_csv_seed3 = QPushButton("CSV Seed 3")
        seed_layout.addWidget(self.btn_auto_seed)
        seed_layout.addWidget(self.btn_csv_seed2)
        seed_layout.addWidget(self.btn_csv_seed3)
        action_layout.addLayout(seed_layout)
        
        save_layout = QHBoxLayout()
        self.btn_save_spine = QPushButton("Spine (z)")
        self.btn_save_sub = QPushButton("Sub (c)")
        self.btn_save_filo = QPushButton("Filo (x)")
        save_layout.addWidget(self.btn_save_spine)
        save_layout.addWidget(self.btn_save_sub)
        save_layout.addWidget(self.btn_save_filo)
        action_layout.addLayout(save_layout)
        
        self.btn_undo = QPushButton("Undo Last")
        action_layout.addWidget(self.btn_undo)
        
        nav_layout = QHBoxLayout()
        self.btn_reset_view = QPushButton("Reset View (a)")
        self.btn_zoom = QPushButton("Zoom Rect (f)")
        self.btn_pan = QPushButton("Pan Image (d)")
        nav_layout.addWidget(self.btn_reset_view)
        nav_layout.addWidget(self.btn_zoom)
        nav_layout.addWidget(self.btn_pan)
        action_layout.addLayout(nav_layout)
        
        self.btn_analyze = QPushButton("Analyze & Save Targets")
        self.btn_analyze.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        action_layout.addWidget(self.btn_analyze)
        
        nav_img_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev Image")
        self.btn_prev.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        self.btn_next = QPushButton("Next Image >")
        self.btn_next.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        nav_img_layout.addWidget(self.btn_prev)
        nav_img_layout.addWidget(self.btn_next)
        action_layout.addLayout(nav_img_layout)
        
        self.input_custom_id = QLineEdit()
        self.input_custom_id.setPlaceholderText("Override Start ID...")
        action_layout.addWidget(self.input_custom_id)
        
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        left_layout.addStretch()
        
        # --- CENTER COLUMN (Matplotlib Canvas) ---
        center_layout = QVBoxLayout()
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8.5, 5.5), sharex=True, sharey=True)
        self.fig.subplots_adjust(bottom=0.05, top=0.92, left=0.02, right=0.98, wspace=0.05)
        
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.StrongFocus) 
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.hide()
        
        self.pink_cmap = ListedColormap(['#ff69b4'])
        self.green_cmap = ListedColormap(['#00ff00'])
        dummy_img = np.zeros((10, 10))
        
        self.img_display = self.ax1.imshow(dummy_img, cmap='gray')
        self.barrier_display = self.ax1.imshow(dummy_img, cmap=self.pink_cmap, alpha=0.3)
        self.mask_display = self.ax1.imshow(dummy_img, cmap=self.green_cmap, alpha=0.6)
        self.click_marker, = self.ax1.plot([], [], 'ro', markersize=4) 
        self.target_marker, = self.ax1.plot([], [], 'c+', markersize=15, markeredgewidth=2) 
        
        self.brush_circle_ax1 = plt.Circle((0, 0), 1, color='cyan', fill=False, linewidth=1.5, visible=False)
        self.brush_circle_ax2 = plt.Circle((0, 0), 1, color='cyan', fill=False, linewidth=1.5, visible=False)
        self.ax1.add_patch(self.brush_circle_ax1)
        self.ax2.add_patch(self.brush_circle_ax2)
        self.ax1.set_title("Z-Slice Navigator")
        
        self.mip_display = self.ax2.imshow(dummy_img, cmap='gray')
        self.mip_barrier_display = self.ax2.imshow(dummy_img, cmap=self.pink_cmap, alpha=0.2)
        self.mip_mask_display = self.ax2.imshow(dummy_img, cmap=self.green_cmap, alpha=0.6)
        self.click_marker_mip, = self.ax2.plot([], [], 'ro', markersize=4)
        self.target_marker_mip, = self.ax2.plot([], [], 'c+', markersize=15, markeredgewidth=2)
        self.ax2.set_title("Global MIP (L-Click: Select | R-Click: Remove)")
        
        center_layout.addWidget(self.canvas)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(120)
        center_layout.addWidget(self.log_output)
        
        # --- RIGHT COLUMN (Controls) ---
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        settings_group = QGroupBox("Processing Settings")
        set_lyt = QVBoxLayout()
        
        set_lyt.addWidget(QLabel("Mask Source:"))
        
        mask_h_lyt = QHBoxLayout()
        self.combo_mask_source = QComboBox()
        self.combo_mask_source.addItems(['SWC Mask', 'Geo Mask', 'Respan Mask'])
        self.combo_mask_source.setCurrentText('Geo Mask')
        mask_h_lyt.addWidget(self.combo_mask_source)
        
        self.btn_auto_barrier = QPushButton("Auto-Gen Barrier")
        self.btn_auto_barrier.setStyleSheet("background-color: #6A1B9A; color: white;")
        mask_h_lyt.addWidget(self.btn_auto_barrier)
        set_lyt.addLayout(mask_h_lyt)
        
        mode_lyt = QHBoxLayout()
        self.bg_mode = QButtonGroup()
        self.rb_target = QRadioButton("Target Spines")
        self.rb_paint = QRadioButton("Paint Barrier")
        self.rb_erase = QRadioButton("Erase Barrier")
        self.rb_target.setChecked(True)
        self.bg_mode.addButton(self.rb_target)
        self.bg_mode.addButton(self.rb_paint)
        self.bg_mode.addButton(self.rb_erase)
        mode_lyt.addWidget(self.rb_target)
        mode_lyt.addWidget(self.rb_paint)
        mode_lyt.addWidget(self.rb_erase)
        set_lyt.addLayout(mode_lyt)
        
        # --- NEW: Snapping Toggle & Radius Slider ---
        self.chk_snap = QCheckBox("Auto-Snap Click to Peak")
        self.chk_snap.setChecked(False) 
        set_lyt.addWidget(self.chk_snap)
        
        self.sld_snap, self.lbl_snap = self.create_slider("Snap Radius (px)", 1, 30, 5, 1)
        set_lyt.addLayout(self.sld_snap)
        # --------------------------------------------

        self.sld_brush, self.lbl_brush = self.create_slider("Brush Size", 2, 40, 8, 1)
        paint_btn_lyt = QHBoxLayout()
        self.btn_clear_paint = QPushButton("Clear Paint")
        self.btn_save_barrier = QPushButton("Save Barrier")
        paint_btn_lyt.addWidget(self.btn_clear_paint)
        paint_btn_lyt.addWidget(self.btn_save_barrier)
        set_lyt.addLayout(self.sld_brush)
        set_lyt.addLayout(paint_btn_lyt)
        
        self.sld_z, self.lbl_z = self.create_slider("Z-Slice", 0, 10, 0, 1)
        self.sld_wl_min, self.lbl_wl_min = self.create_slider("Contrast Min", 0, 65535, 0, 1)
        self.sld_wl_max, self.lbl_wl_max = self.create_slider("Contrast Max", 0, 65535, 65535, 1)
        
        self.sld_barrier, self.lbl_barrier = self.create_slider("Barrier µm", 0, 40, 12, 10.0)
        self.sld_tol, self.lbl_tol = self.create_slider("Tolerance", 5, 90, 45, 100.0)
        self.sld_zsearch, self.lbl_zsearch = self.create_slider("Z-Search", 0, 20, 10, 1)
        self.sld_max_geo, self.lbl_max_geo = self.create_slider("Max Geodesic µm", 10, 150, 50, 10.0)
        
        for s in [self.sld_z, self.sld_wl_min, self.sld_wl_max, self.sld_barrier, self.sld_tol, self.sld_zsearch, self.sld_max_geo]:
            set_lyt.addLayout(s)
            
        settings_group.setLayout(set_lyt)
        right_layout.addWidget(settings_group)
        
        topo_group = QGroupBox("Topological Restraints")
        topo_lyt = QVBoxLayout()
        self.chk_hessian = QCheckBox("Use 2D Curvature Isolation")
        self.chk_hessian.setChecked(True)
        topo_lyt.addWidget(self.chk_hessian)
        
        self.sld_strict, self.lbl_strict = self.create_slider("Blob Strictness", -50, 50, 0, 1000.0)
        self.sld_sigma, self.lbl_sigma = self.create_slider("Hessian σ", 5, 40, 15, 10.0)
        topo_lyt.addLayout(self.sld_strict)
        topo_lyt.addLayout(self.sld_sigma)
        topo_group.setLayout(topo_lyt)
        right_layout.addWidget(topo_group)
        
        disp_group = QGroupBox("Display")
        disp_lyt = QHBoxLayout()
        self.chk_barrier = QCheckBox("Show Barrier")
        self.chk_markers = QCheckBox("Show Markers")
        self.chk_segment = QCheckBox("Show Segment")
        self.chk_barrier.setChecked(True)
        self.chk_markers.setChecked(True)
        self.chk_segment.setChecked(True)
        disp_lyt.addWidget(self.chk_barrier)
        disp_lyt.addWidget(self.chk_markers)
        disp_lyt.addWidget(self.chk_segment)
        disp_group.setLayout(disp_lyt)
        right_layout.addWidget(disp_group)
        right_layout.addStretch()
        
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(center_layout, 6)
        main_layout.addLayout(right_layout, 2)
        
        self.connect_signals()

    def create_slider(self, label, min_val, max_val, default, divisor=1):
        lyt = QHBoxLayout()
        lbl_name = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        lbl_val = QLabel(str(default / divisor))
        slider.setProperty('divisor', divisor)
        slider.valueChanged.connect(lambda v, l=lbl_val, d=divisor: l.setText(str(v/d)))
        lyt.addWidget(lbl_name)
        lyt.addWidget(slider)
        lyt.addWidget(lbl_val)
        return lyt, slider

    def apply_dark_theme(self):
        dark_stylesheet = """
        QMainWindow { background-color: #1e1e1e; color: #ffffff; }
        QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
        QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #aaa; }
        QPushButton { background-color: #333; border: 1px solid #555; padding: 6px; border-radius: 4px; color: white; }
        QPushButton:hover { background-color: #444; border: 1px solid #888; }
        QLineEdit, QComboBox, QListWidget, QTextEdit { background-color: #2d2d2d; border: 1px solid #555; color: white; border-radius: 3px; padding: 4px;}
        QSlider::groove:horizontal { border: 1px solid #3A3939; height: 8px; background: #201F1F; border-radius: 4px; }
        QSlider::handle:horizontal { background: #007ACC; border: 1px solid #005A99; width: 14px; margin: -3px 0; border-radius: 7px; }
        QCheckBox { spacing: 5px; }
        QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555; border-radius: 3px; background: #2d2d2d; }
        QCheckBox::indicator:checked { background: #007ACC; }
        """
        self.setStyleSheet(dark_stylesheet)

    def log(self, msg, clear=False):
        if clear: self.log_output.clear()
        self.log_output.append(msg)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
        QApplication.processEvents() 

    # ==========================================
    # 3. LOGIC & EVENT CONNECTIONS
    # ==========================================
    def connect_signals(self):
        QShortcut(QKeySequence("z"), self).activated.connect(lambda: self.on_save_target('spine'))
        QShortcut(QKeySequence("x"), self).activated.connect(lambda: self.on_save_target('filopodia'))
        QShortcut(QKeySequence("c"), self).activated.connect(lambda: self.on_save_target('suboptimal'))
        QShortcut(QKeySequence("u"), self).activated.connect(self.on_undo_target)
        QShortcut(QKeySequence("Delete"), self).activated.connect(self.on_delete_selected_target)
        QShortcut(QKeySequence("Backspace"), self).activated.connect(self.on_delete_selected_target)
        QShortcut(QKeySequence("a"), self).activated.connect(self.toolbar.home)
        QShortcut(QKeySequence("f"), self).activated.connect(self.toolbar.zoom)
        QShortcut(QKeySequence("d"), self).activated.connect(self.toolbar.pan)

        self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_load.clicked.connect(self.on_load_folder)
        self.btn_load_analyzed.clicked.connect(self.on_load_analyzed_folder)
        self.btn_batch.clicked.connect(self.on_batch_unattended)
        
        self.combo_mask_source.currentTextChanged.connect(self.on_mask_source_change)
        self.btn_auto_barrier.clicked.connect(self.on_auto_generate_barrier) 
        
        self.rb_target.toggled.connect(self.refresh_display)
        self.btn_clear_paint.clicked.connect(self.on_clear_paint)
        self.btn_save_barrier.clicked.connect(self.on_save_barrier)
        
        self.lbl_z.valueChanged.connect(self.on_z_scroll_sync)
        self.lbl_wl_min.valueChanged.connect(self.update_contrast)
        self.lbl_wl_max.valueChanged.connect(self.update_contrast)
        self.lbl_barrier.valueChanged.connect(self.on_barrier_change)
        
        self.chk_barrier.toggled.connect(self.refresh_display)
        self.chk_markers.toggled.connect(self.refresh_display)
        self.chk_segment.toggled.connect(self.refresh_display)
        
        self.list_targets.itemSelectionChanged.connect(self.on_target_selected)
        self.btn_delete_target.clicked.connect(self.on_delete_selected_target)
        self.btn_apply_status.clicked.connect(self.on_apply_status)
        self.btn_rename.clicked.connect(self.on_rename_target)
        
        self.btn_auto_seed.clicked.connect(lambda: self.auto_generate_seeds())
        self.btn_csv_seed2.clicked.connect(lambda: self.load_csv_seeds('seed2'))
        self.btn_csv_seed3.clicked.connect(lambda: self.load_csv_seeds('seed3'))
        
        self.btn_save_spine.clicked.connect(lambda: self.on_save_target('spine'))
        self.btn_save_sub.clicked.connect(lambda: self.on_save_target('suboptimal'))
        self.btn_save_filo.clicked.connect(lambda: self.on_save_target('filopodia'))
        self.btn_undo.clicked.connect(self.on_undo_target)
        
        self.btn_reset_view.clicked.connect(self.toolbar.home)
        self.btn_zoom.clicked.connect(self.toolbar.zoom)
        self.btn_pan.clicked.connect(self.toolbar.pan)
        
        self.btn_analyze.clicked.connect(lambda: self.on_analyze_all())
        self.btn_prev.clicked.connect(self.on_prev_image)
        self.btn_next.clicked.connect(self.on_next_image)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def val(self, slider):
        return slider.value() / slider.property('divisor')

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder = folder
            self.lbl_file_info.setText(f"Selected: {os.path.basename(folder)}")

    def update_contrast(self):
        if self.raw_stack is not None:
            self.img_display.set_clim(self.lbl_wl_min.value(), self.lbl_wl_max.value())
            self.mip_display.set_clim(self.lbl_wl_min.value(), self.lbl_wl_max.value())
            self.fig.canvas.draw_idle()

    def on_z_scroll_sync(self):
        self.z = self.lbl_z.value()
        self.refresh_display()

    def get_effective_barrier(self):
        base = self.shaft_barrier if self.shaft_barrier is not None else np.zeros_like(self.raw_stack, dtype=bool)
        painted_3d = np.broadcast_to(self.painted_barrier_2d, base.shape) if self.painted_barrier_2d is not None else np.zeros_like(base)
        erased_3d = np.broadcast_to(self.erased_barrier_2d, base.shape) if self.erased_barrier_2d is not None else np.zeros_like(base)
        return (base | painted_3d) & (~erased_3d)

    def get_2d_hessian_blob_mask(self, V_sub, strictness, sig):
        mask = np.zeros_like(V_sub, dtype=bool)
        for zi in range(V_sub.shape[0]):
            slice_v = V_sub[zi].astype(float)
            if slice_v.max() > 0: slice_v = slice_v / slice_v.max()
            else: continue
                
            Dyy = gaussian_filter(slice_v, sigma=sig, order=[2, 0])
            Dxx = gaussian_filter(slice_v, sigma=sig, order=[0, 2])
            Dxy = gaussian_filter(slice_v, sigma=sig, order=[1, 1])
            
            H2 = np.zeros((2, 2) + slice_v.shape)
            H2[0,0]=Dyy; H2[0,1]=Dxy; H2[1,0]=Dxy; H2[1,1]=Dxx
            H2 = np.moveaxis(H2, [0,1], [-2,-1])
            
            l1 = np.linalg.eigvalsh(H2)[..., 1]
            mask[zi] = l1 <= strictness
        return mask

    def process_swc_file(self, stack_shape, swc_file, raw_stack):
        swc_data = pd.read_csv(swc_file, sep=r'\s+', comment='#', header=None, names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'])
        skeleton_2d = np.zeros((stack_shape[1], stack_shape[2]), dtype=bool)
        for _, row in swc_data.iterrows():
            y_idx = int(round(row['y']/self.dy))
            x_idx = int(round(row['x']/self.dx))
            if (0 <= y_idx < stack_shape[1] and 0 <= x_idx < stack_shape[2]):
                skeleton_2d[y_idx, x_idx] = True

        distance_field_2d = distance_transform_edt(~skeleton_2d, sampling=[self.dy, self.dx])
        dist_3d = np.broadcast_to(distance_field_2d, stack_shape).copy()
        
        swc_dict = swc_data.set_index('id').to_dict('index')
        total_length = 0.0
        for node_id, data in swc_dict.items():
            parent_id = data['parent']
            if parent_id in swc_dict:
                parent_data = swc_dict[parent_id]
                dist = np.sqrt((data['x'] - parent_data['x'])**2 + (data['y'] - parent_data['y'])**2 + (data['z'] - parent_data['z'])**2)
                total_length += dist
        return dist_3d, total_length

    def on_auto_generate_barrier(self):
        if self.raw_stack is None:
            self.log("⚠️ No image loaded to generate barrier.", clear=True)
            return
            
        self.log("⚙️ Auto-generating dendritic barrier using geodesic reconstruction...", clear=True)
        QApplication.processEvents()

        mip = np.max(self.raw_stack, axis=0) if self.raw_stack.ndim == 3 else self.raw_stack
        smoothed = gaussian(mip.astype(float), sigma=1.0)
        
        try:
            thresh = filters.threshold_otsu(smoothed)
        except Exception as e:
            self.log(f"❌ Thresholding failed: {e}")
            return
            
        binary_mask = smoothed > thresh
        labeled = measure.label(binary_mask)
        if labeled.max() == 0:
            self.log("❌ Failed to isolate a dendrite. Try manual masking.")
            return
            
        largest_cc = (labeled == np.argmax(np.bincount(labeled.flat)[1:]) + 1)
        edt_2d = distance_transform_edt(largest_cc)
        skel = morphology.skeletonize(largest_cc)
        
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbor_count = convolve(skel.astype(int), kernel, mode='constant')
        endpoints = np.argwhere((neighbor_count == 11) & skel)
        
        if len(endpoints) < 2:
            self.log("❌ Could not find reliable skeleton endpoints.")
            return
            
        D = squareform(pdist(endpoints))
        i, j = np.unravel_index(np.argmax(D), D.shape)
        start_pt, end_pt = tuple(endpoints[i]), tuple(endpoints[j])
        
        cost_map = np.where(skel, 1, 1e6)
        path, _ = graph.route_through_array(cost_map, start=start_pt, end=end_pt, fully_connected=True)
        
        reconstructed_shaft = np.zeros_like(largest_cc)
        for r, c in path:
            radius = edt_2d[r, c]
            if radius > 0:
                rr, cc = disk((r, c), radius + 0.5, shape=reconstructed_shaft.shape)
                reconstructed_shaft[rr, cc] = True
                
        dir_name = os.path.dirname(self.files[self.current_idx])
        base_name = os.path.splitext(os.path.basename(self.files[self.current_idx]))[0]
        if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        
        geo_path = os.path.join(dir_name, base_name + '_dendrite-geo.tif')
        imwrite(geo_path, (reconstructed_shaft.astype(np.uint8) * 255))
        
        self.dendrite_length_um = len(path) * self.dy  
        self.log(f"✅ Auto-Barrier generated & saved to {os.path.basename(geo_path)}! Length: ~{self.dendrite_length_um:.2f} µm.")
        
        self.combo_mask_source.blockSignals(True)
        self.combo_mask_source.setCurrentText('Geo Mask')
        self.combo_mask_source.blockSignals(False)
        self.lbl_barrier.setValue(0)
        
        self.apply_mask_source()
        self.refresh_display()

    def apply_mask_source(self):
        if self.raw_stack is None: return
        filepath = self.files[self.current_idx]
        dir_name = os.path.dirname(filepath)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        
        val = self.combo_mask_source.currentText()
        dist_3d = np.inf * np.ones_like(self.raw_stack)
        d_len = 0.0
        
        if val == 'SWC Mask':
            swc_path = filepath.replace('.tif', '.swc').replace('.tiff', '.swc')
            if os.path.exists(swc_path):
                dist_3d, d_len = self.process_swc_file(self.raw_stack.shape, swc_path, self.raw_stack)
                self.log(f"✅ Found SWC! Length: {d_len:.2f} µm")
            else:
                self.log("⚠️ No SWC file found. Default barrier disabled.")
        else:
            pattern = "*geo*.tif*" if val == 'Geo Mask' else "*respan*.tif*"
            matches = glob(os.path.join(dir_name, f"{base_name}{pattern}"))
            
            if matches:
                mask_file = matches[0]
                self.log(f"✅ Found {val} file: {os.path.basename(mask_file)}")
                loaded_mask = imread(mask_file) > 0
                
                if loaded_mask.shape[-2:] != self.raw_stack.shape[-2:]:
                    self.log(f"⚠️ Mask shape {loaded_mask.shape} mismatches image {self.raw_stack.shape}. Ignoring mask.")
                else:
                    if loaded_mask.ndim == 2:
                        dist_2d = distance_transform_edt(~loaded_mask, sampling=[self.dy, self.dx])
                        dist_3d = np.broadcast_to(dist_2d, self.raw_stack.shape).copy()
                    elif loaded_mask.ndim == 3:
                        dist_3d = distance_transform_edt(~loaded_mask, sampling=[self.dz, self.dy, self.dx])
            else:
                self.log(f"⚠️ No {val} file found matching '{base_name}{pattern}'.")
                
        self.dist_field_3d = dist_3d
        self.dendrite_length_um = d_len
        self.initial_shaft_barrier = self.dist_field_3d <= self.val(self.lbl_barrier)
        self.avg_initial_dendrite_intensity = float(np.mean(self.raw_stack[self.initial_shaft_barrier])) if np.any(self.initial_shaft_barrier) else 0.0
        self.shaft_barrier = self.dist_field_3d <= self.val(self.lbl_barrier)

    def on_mask_source_change(self, val):
        if val in ['Geo Mask', 'Respan Mask']:
            self.lbl_barrier.setValue(0)
        if self.raw_stack is not None:
            self.apply_mask_source()
            self.refresh_display()

    def find_optimal_xyz(self, x, y, search_radius=5, start_z=None):
        if self.base_smoothed_stack is None: return self.z if start_z is None else start_z, y, x
        stack = self.base_smoothed_stack
        y_min, y_max = max(0, y - search_radius), min(stack.shape[1], y + search_radius + 1)
        x_min, x_max = max(0, x - search_radius), min(stack.shape[2], x + search_radius + 1)
        
        if start_z is None:
            z_profile = stack[:, y, x]
            eff_barrier_1d = self.get_effective_barrier()[:, y, x]
            masked_profile = np.where(eff_barrier_1d, -1e9, z_profile)
            best_z = int(np.argmax(z_profile)) if np.all(masked_profile == -1e9) else int(np.argmax(masked_profile))
        else:
            best_z = start_z
            
        z_min, z_max = max(0, best_z - 1), min(stack.shape[0], best_z + 2)
        sub_volume = stack[z_min:z_max, y_min:y_max, x_min:x_max].copy().astype(np.float64)
        sub_volume = uniform_filter(sub_volume, size=3)
        eff_bar_sub = self.get_effective_barrier()[z_min:z_max, y_min:y_max, x_min:x_max]
        sub_volume[eff_bar_sub] = -1e9
                
        if np.all(sub_volume == -1e9): return best_z, y, x 
        max_idx = np.argmax(sub_volume)
        z_loc, dy_loc, dx_loc = np.unravel_index(max_idx, sub_volume.shape)
        return int(z_min + z_loc), int(y_min + dy_loc), int(x_min + dx_loc)

    def load_image_at_index(self, idx):
        if not self.files or idx >= len(self.files): return
        filepath = self.files[idx]
        filename = os.path.basename(filepath)
        self.lbl_file_info.setText(f"Remaining {idx+1}/{len(self.files)}: {filename}")
        
        in_path = self.input_folder if self.input_folder else os.path.dirname(filepath)
        out_dir = os.path.join(in_path, 'output_analysis')
        base_name = os.path.splitext(filename)[0]
        if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        
        mode_str = "Correction Mode" if self.is_correction_mode else "Standard Mode"
        self.log(f"Loading {filename} [{mode_str}]...", clear=True)
        
        self.raw_stack = imread(filepath)
        if self.raw_stack.ndim == 2:
            self.raw_stack = np.expand_dims(self.raw_stack, axis=0)

        self.base_smoothed_stack = gaussian(self.raw_stack, sigma=1.0, preserve_range=True).astype(self.raw_stack.dtype)
        
        h, w = self.raw_stack.shape[1], self.raw_stack.shape[2]
        self.ax1.set_xlim(-0.5, w - 0.5); self.ax1.set_ylim(h - 0.5, -0.5)
        
        extent = [-0.5, w - 0.5, h - 0.5, -0.5]
        for disp in [self.img_display, self.barrier_display, self.mask_display, self.mip_display, self.mip_barrier_display, self.mip_mask_display]:
            disp.set_extent(extent)
            
        empty = np.zeros((h, w))
        for disp in [self.barrier_display, self.mip_barrier_display, self.mask_display, self.mip_mask_display]:
            disp.set_data(empty)
            disp.set_alpha(0.0)
        
        self.apply_mask_source()

        custom_barrier_2d_path = os.path.join(out_dir, base_name + '_custom_barrier_2d.tif')
        if os.path.exists(custom_barrier_2d_path):
            edit_mask = imread(custom_barrier_2d_path)
            self.painted_barrier_2d = (edit_mask == 1)
            self.erased_barrier_2d = (edit_mask == 2)
            self.log("📂 Loaded previously saved custom 2D barrier edits!")
        else:
            self.painted_barrier_2d = np.zeros((h, w), dtype=bool)
            self.erased_barrier_2d = np.zeros((h, w), dtype=bool)

        self.z = 0
        self.mask = np.zeros_like(self.raw_stack, dtype=np.uint16)
        self.is_drawing = False
        
        self.click_x = self.click_y = self.target_x = self.target_y = self.target_z = None
        self.saved_targets = []
        self.target_counter = 1
        self.loaded_df = None
        self.input_custom_id.setText('')
        
        if self.is_correction_mode:
            csv_path_corr = os.path.join(out_dir, base_name + '_corrected_spine_results.csv')
            csv_path_orig = os.path.join(out_dir, base_name + '_spine_results.csv')
            mask_path_corr = os.path.join(out_dir, base_name + '_corrected_segmentation_mask.tif')
            mask_path_orig = os.path.join(out_dir, base_name + '_segmentation_mask.tif')
            
            csv_to_load = csv_path_corr if os.path.exists(csv_path_corr) else csv_path_orig
            mask_to_load = mask_path_corr if os.path.exists(mask_path_corr) else mask_path_orig
            
            if os.path.exists(csv_to_load) and os.path.exists(mask_to_load):
                self.log(f"📥 Loading analyzed records from {os.path.basename(csv_to_load)}...")
                df = pd.read_csv(csv_to_load)
                self.loaded_df = df
                loaded_mask = imread(mask_to_load)
                
                if loaded_mask.max() == 255 and len(np.unique(loaded_mask)) <= 2:
                    self.log("⚠️ Legacy binary mask detected. Re-analyze to convert to instance mask for selective deletion.")
                    self.mask = (loaded_mask > 0).astype(np.uint16) 
                else:
                    self.mask = loaded_mask.astype(np.uint16)
                
                max_id = 0
                for _, row in df.iterrows():
                    tid = int(row['Target_ID'])
                    max_id = max(max_id, tid)
                    t_type = 'filopodia' if row['Classification'] == 'Filopodia' else 'suboptimal' if row['Classification'] == 'Suboptimal Measures' else 'spine'
                    prefix = "[Filo]" if t_type == 'filopodia' else "[Sub]" if t_type == 'suboptimal' else ""
                    orig_status = str(row['Status']) if 'Status' in df.columns else 'static'
                    
                    t_dict = {
                        'idx': tid,
                        'z': int(row['Z_Slice']) - 1, 'y': int(row['Corrected_Y']), 'x': int(row['Corrected_X']),
                        'click_x': int(row['Original_X']), 'click_y': int(row['Original_Y']),
                        'target_type': t_type, 'status': orig_status, 'is_modified': False
                    }
                    t_dict['label'] = f"{prefix} [{tid}] Z:{t_dict['z']+1} Y:{t_dict['y']} X:{t_dict['x']} ({orig_status})".strip()
                    self.saved_targets.append(t_dict)
                
                self.target_counter = max_id + 1
        
        self.update_list_ui()
        
        self.lbl_z.blockSignals(True)
        self.lbl_z.setRange(0, self.raw_stack.shape[0] - 1)
        self.lbl_z.setValue(0)
        self.lbl_z.blockSignals(False)
        
        rmin, rmax = int(self.raw_stack.min()), int(self.raw_stack.max())
        self.lbl_wl_min.setRange(rmin, rmax)
        self.lbl_wl_max.setRange(rmin, rmax)
        self.lbl_wl_min.setValue(int(np.percentile(self.raw_stack, 1)))
        self.lbl_wl_max.setValue(int(np.percentile(self.raw_stack, 99)))
        
        mip_raw = np.max(self.raw_stack, axis=0)
        self.mip_display.set_data(mip_raw)
        
        self.refresh_display()

    def update_list_ui(self):
        self.list_targets.clear()
        for t in self.saved_targets:
            self.list_targets.addItem(t['label'])

    def refresh_display(self):
        if self.raw_stack is None: return
        z = self.z
        self.img_display.set_data(self.raw_stack[z])
        self.update_contrast()
        
        if self.chk_barrier.isChecked():
            eff_barrier = self.get_effective_barrier()
            barrier_z = eff_barrier[z]
            barrier_mip = np.max(eff_barrier, axis=0)
            
            if np.any(barrier_z):
                self.barrier_display.set_data(np.ma.masked_where(~barrier_z, barrier_z))
                self.barrier_display.set_alpha(0.3)
            else:
                self.barrier_display.set_data(np.zeros_like(barrier_z))
                self.barrier_display.set_alpha(0.0)
                
            if np.any(barrier_mip):
                self.mip_barrier_display.set_data(np.ma.masked_where(~barrier_mip, barrier_mip))
                self.mip_barrier_display.set_alpha(0.2)
            else:
                self.mip_barrier_display.set_data(np.zeros_like(barrier_mip))
                self.mip_barrier_display.set_alpha(0.0)
        else:
            self.barrier_display.set_data(np.zeros_like(self.raw_stack[z]))
            self.barrier_display.set_alpha(0.0)
            self.mip_barrier_display.set_data(np.zeros_like(self.raw_stack[0]))
            self.mip_barrier_display.set_alpha(0.0)
        
        if self.chk_segment.isChecked():
            z_mask = self.mask[z] > 0
            mip_m = np.max(self.mask, axis=0) > 0
            
            if np.any(z_mask):
                self.mask_display.set_data(np.ma.masked_where(~z_mask, z_mask))
                self.mask_display.set_alpha(0.6)
            else:
                self.mask_display.set_data(np.zeros_like(z_mask))
                self.mask_display.set_alpha(0.0)
                
            if np.any(mip_m):
                self.mip_mask_display.set_data(np.ma.masked_where(~mip_m, mip_m))
                self.mip_mask_display.set_alpha(0.6)
            else:
                self.mip_mask_display.set_data(np.zeros_like(mip_m))
                self.mip_mask_display.set_alpha(0.0)
        else:
            self.mask_display.set_data(np.zeros_like(self.raw_stack[z]))
            self.mask_display.set_alpha(0.0)
            self.mip_mask_display.set_data(np.zeros_like(self.raw_stack[0]))
            self.mip_mask_display.set_alpha(0.0)
        
        for item in self.texts_ax1 + self.texts_ax2 + self.dots_ax1 + self.dots_ax2:
            try: item.remove()
            except: pass
        self.texts_ax1.clear(); self.texts_ax2.clear(); self.dots_ax1.clear(); self.dots_ax2.clear()
        
        if self.chk_markers.isChecked():
            if self.click_x is not None and self.rb_target.isChecked():
                self.click_marker.set_data([self.click_x], [self.click_y])
                self.click_marker_mip.set_data([self.click_x], [self.click_y])
            else:
                self.click_marker.set_data([], []); self.click_marker_mip.set_data([], [])

            if self.target_z is not None and self.target_x is not None and self.rb_target.isChecked():
                self.target_marker.set_data([self.target_x], [self.target_y])
                self.target_marker_mip.set_data([self.target_x], [self.target_y])
            else:
                self.target_marker.set_data([], []); self.target_marker_mip.set_data([], [])
                
            for t in self.saved_targets:
                color = 'blue' if t.get('target_type') == 'filopodia' else 'orange' if t.get('target_type') == 'suboptimal' else 'red'
                cx, cy = t.get('click_x', t['x']), t.get('click_y', t['y'])
                d1_c, = self.ax1.plot(cx, cy, marker='o', color='red', markersize=3, linestyle='None')
                d2_c, = self.ax2.plot(cx, cy, marker='o', color='red', markersize=3, linestyle='None')
                self.dots_ax1.append(d1_c); self.dots_ax2.append(d2_c)
                
                txt1 = self.ax1.text(cx + 3, cy, str(t['idx']), color=color, fontsize=10, fontweight='bold')
                txt2 = self.ax2.text(cx + 3, cy, str(t['idx']), color=color, fontsize=10, fontweight='bold')
                self.texts_ax1.append(txt1); self.texts_ax2.append(txt2)
        else:
            self.click_marker.set_data([], []); self.click_marker_mip.set_data([], [])
            self.target_marker.set_data([], []); self.target_marker_mip.set_data([], [])
        
        self.lbl_z.blockSignals(True)
        self.lbl_z.setValue(z)
        self.lbl_z.blockSignals(False)
        self.fig.canvas.draw_idle()

    def paint(self, event):
        if not self.is_drawing or self.raw_stack is None: return
        if event.inaxes not in [self.ax1, self.ax2]: return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        rr, cc = disk((y, x), int(self.val(self.lbl_brush)), shape=self.painted_barrier_2d.shape)
        
        if self.rb_paint.isChecked():
            self.painted_barrier_2d[rr, cc] = True; self.erased_barrier_2d[rr, cc] = False
        elif self.rb_erase.isChecked():
            self.erased_barrier_2d[rr, cc] = True; self.painted_barrier_2d[rr, cc] = False
        
        if self.chk_barrier.isChecked():
            eff_barrier = self.get_effective_barrier()
            self.barrier_display.set_data(np.ma.masked_where(~eff_barrier[self.z], eff_barrier[self.z]))
            self.barrier_display.set_alpha(0.3)
            self.mip_barrier_display.set_data(np.ma.masked_where(~np.max(eff_barrier, axis=0), np.max(eff_barrier, axis=0)))
            self.mip_barrier_display.set_alpha(0.2)
            self.fig.canvas.draw_idle()

    def on_mouse_press(self, event):
        if self.toolbar.mode != '': return
        if self.rb_paint.isChecked() or self.rb_erase.isChecked():
            self.is_drawing = True; self.paint(event)
        elif self.rb_target.isChecked() and event.inaxes in [self.ax1, self.ax2]:
            clicked_x, clicked_y = int(round(event.xdata)), int(round(event.ydata))
            if event.button == 3: 
                min_dist, to_remove = 12, None
                for t in self.saved_targets:
                    dist = np.hypot(t['click_x'] - clicked_x, t['click_y'] - clicked_y)
                    if dist < min_dist: min_dist, to_remove = dist, t
                if to_remove:
                    self.mask[self.mask == to_remove['idx']] = 0
                    self.saved_targets.remove(to_remove)
                    self.update_list_ui()
                    self.refresh_display()
                    self.log(f"🗑️ Removed Target [{to_remove['idx']}] at X:{to_remove['click_x']}, Y:{to_remove['click_y']}", clear=True)
                return
            if event.button == 1:  
                self.click_x, self.click_y = clicked_x, clicked_y
                
                search_rad = int(self.val(self.lbl_snap)) if self.chk_snap.isChecked() else 0
                opt_z, opt_y, opt_x = self.find_optimal_xyz(clicked_x, clicked_y, search_radius=search_rad, start_z=self.z)
                
                self.target_z, self.target_y, self.target_x = opt_z, opt_y, opt_x
                self.z = opt_z
                self.refresh_display()

    def on_mouse_motion(self, event):
        if event.inaxes in [self.ax1, self.ax2] and event.xdata is not None and event.ydata is not None:
            r = int(self.val(self.lbl_brush))
            if self.rb_paint.isChecked() or self.rb_erase.isChecked():
                self.brush_circle_ax1.center = (event.xdata, event.ydata)
                self.brush_circle_ax1.set_radius(r)
                self.brush_circle_ax1.set_visible(event.inaxes == self.ax1)
                self.brush_circle_ax2.center = (event.xdata, event.ydata)
                self.brush_circle_ax2.set_radius(r)
                self.brush_circle_ax2.set_visible(event.inaxes == self.ax2)
                self.fig.canvas.draw_idle()
            else:
                self.brush_circle_ax1.set_visible(False)
                self.brush_circle_ax2.set_visible(False)
        self.paint(event)

    def on_mouse_release(self, event): self.is_drawing = False

    def on_scroll(self, event):
        if event.inaxes not in [self.ax1, self.ax2] or self.raw_stack is None: return
        if event.button == 'up': self.z = min(self.z + 1, self.raw_stack.shape[0] - 1)
        elif event.button == 'down': self.z = max(self.z - 1, 0)
        self.refresh_display()

    def on_clear_paint(self):
        if self.raw_stack is not None:
            self.painted_barrier_2d = np.zeros_like(self.painted_barrier_2d)
            self.erased_barrier_2d = np.zeros_like(self.erased_barrier_2d)
            self.refresh_display()

    def on_save_barrier(self):
        if self.raw_stack is None: return
        in_path = self.input_folder if self.input_folder else os.path.dirname(self.files[self.current_idx])
        out_dir = os.path.join(in_path, 'output_analysis')
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.files[self.current_idx]))[0]
        if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        barrier_img_path = os.path.join(out_dir, base_name + '_custom_barrier_2d.tif')
        
        edit_mask = np.zeros((self.raw_stack.shape[1], self.raw_stack.shape[2]), dtype=np.uint8)
        edit_mask[self.painted_barrier_2d] = 1; edit_mask[self.erased_barrier_2d] = 2
        imwrite(barrier_img_path, edit_mask)
        self.log(f"💾 Custom 2D barrier saved to {base_name}_custom_barrier_2d.tif")

    def on_barrier_change(self):
        if self.dist_field_3d is not None:
            self.shaft_barrier = self.dist_field_3d <= self.val(self.lbl_barrier)
            self.refresh_display()

    def on_save_target(self, target_type='spine'):
        if self.click_x is None:
            self.log("⚠️ Please Left-Click on the image to set a spatial target location before saving.", clear=True)
            return
        if not self.rb_target.isChecked():
            self.log("⚠️ Switch mode to 'Target Spines' to save targets.", clear=True)
            return

        custom_val = self.input_custom_id.text().strip()
        if custom_val.isdigit():
            self.target_counter = int(custom_val)
            self.input_custom_id.setText('')
            
        idx = self.target_counter
        prefix = "[Filo]" if target_type == 'filopodia' else "[Sub]" if target_type == 'suboptimal' else ""
        status = 'new'
        label_text = f"{prefix} [{idx}] Z:{self.target_z+1} Y:{self.target_y} X:{self.target_x} ({status})".strip()
        
        if not any(t['z'] == self.target_z and t['y'] == self.target_y and t['x'] == self.target_x for t in self.saved_targets):
            self.saved_targets.append({'idx': idx, 'label': label_text, 'z': self.target_z, 'y': self.target_y, 'x': self.target_x,
                                       'click_x': self.click_x, 'click_y': self.click_y, 'target_type': target_type, 'status': status, 'is_modified': True})
            self.target_counter += 1
            self.update_list_ui()
            tag = "Filopodia" if target_type == 'filopodia' else "Suboptimal Spine" if target_type == 'suboptimal' else "Target Spine"
            self.log(f"💾 Added {tag} {idx} at Z:{self.target_z+1}, Y:{self.target_y}, X:{self.target_x} ({status})", clear=True)
            self.refresh_display()
        else:
            self.log(f"⚠️ Target already exists exactly at Z:{self.target_z+1}, Y:{self.target_y}, X:{self.target_x}.", clear=True)

    def on_delete_selected_target(self):
        selected_items = self.list_targets.selectedItems()
        if selected_items and self.saved_targets:
            removed_count = 0
            for item in selected_items:
                sel_label = item.text()
                target_data = next((t for t in self.saved_targets if t['label'] == sel_label), None)
                if target_data: 
                    self.mask[self.mask == target_data['idx']] = 0
                    self.saved_targets.remove(target_data)
                    removed_count += 1
            self.update_list_ui()
            self.refresh_display()
            self.log(f"🗑️ Deleted {removed_count} Target(s) from list.", clear=True)

    def on_rename_target(self):
        selected = self.list_targets.selectedItems()
        new_id_str = self.input_rename.text().strip()
        if len(selected) != 1:
            self.log("⚠️ Please select exactly ONE target to update its ID.", clear=True)
            return
        if not new_id_str.isdigit():
            self.log("⚠️ Please enter a valid number for the new ID.", clear=True)
            return
            
        new_id = int(new_id_str)
        target_data = next((t for t in self.saved_targets if t['label'] == selected[0].text()), None)
        if target_data:
            old_id = target_data['idx']
            target_data['idx'] = new_id
            self.mask[self.mask == old_id] = new_id
            
            prefix = "[Filo]" if target_data.get('target_type') == 'filopodia' else "[Sub]" if target_data.get('target_type') == 'suboptimal' else ""
            status = target_data.get('status', 'new')
            new_label = f"{prefix} [{new_id}] Z:{target_data['z']+1} Y:{target_data['y']} X:{target_data['x']} ({status})".strip()
            target_data['label'] = new_label
            target_data['is_modified'] = True
            self.update_list_ui()
            self.input_rename.setText('')
            self.refresh_display()
            self.log(f"✏️ Updated target to ID [{new_id}]", clear=True)

    def on_apply_status(self):
        selected = self.list_targets.selectedItems()
        new_status = self.combo_status.currentText()
        if not selected:
            self.log("⚠️ Please select at least one target to apply status.", clear=True)
            return
        for item in selected:
            target_data = next((t for t in self.saved_targets if t['label'] == item.text()), None)
            if target_data:
                target_data['status'] = new_status
                prefix = "[Filo]" if target_data.get('target_type') == 'filopodia' else "[Sub]" if target_data.get('target_type') == 'suboptimal' else ""
                new_label = f"{prefix} [{target_data['idx']}] Z:{target_data['z']+1} Y:{target_data['y']} X:{target_data['x']} ({new_status})".strip()
                target_data['label'] = new_label
        self.update_list_ui()
        self.refresh_display()
        self.log(f"✅ Applied status '{new_status}' to {len(selected)} target(s).", clear=True)

    def on_undo_target(self):
        if self.saved_targets:
            removed = self.saved_targets.pop()
            self.mask[self.mask == removed['idx']] = 0
            self.update_list_ui()
            self.refresh_display()
            self.log(f"↩️ Undid Target [{removed['idx']}]. Remaining: {len(self.saved_targets)}", clear=True)

    def on_target_selected(self):
        selected = self.list_targets.selectedItems()
        if selected:
            target_data = next((t for t in self.saved_targets if t['label'] == selected[0].text()), None)
            if target_data:
                self.target_z, self.target_y, self.target_x = target_data['z'], target_data['y'], target_data['x']
                self.click_x, self.click_y = target_data.get('click_x'), target_data.get('click_y')
                self.z = target_data['z']
                self.rb_target.setChecked(True)
                self.refresh_display()

    # --- AUTOMATIC SEEDERS ---
    def auto_generate_seeds(self, silent=False):
        if self.raw_stack is None: return
        if not silent: self.log("🔍 Running background subtraction & geodesic filtering for auto-seeding...", clear=True)
        
        self.saved_targets = []
        if self.input_custom_id.text().strip().isdigit():
            self.target_counter = int(self.input_custom_id.text().strip())
            self.input_custom_id.setText('')
        elif not self.is_correction_mode: 
            self.target_counter = 1
            
        mip_img = np.max(self.raw_stack, axis=0)
        background = uniform_filter(mip_img.astype(float), size=50)
        bg_subtracted = np.clip(mip_img.astype(float) - background, 0, None)
        smoothed = gaussian_filter(bg_subtracted, sigma=1.5)
        
        if smoothed.max() > 0:
            binary_objects = smoothed > np.percentile(smoothed[smoothed > 0], 88)
            binary_objects[np.max(self.get_effective_barrier(), axis=0)] = False
            labeled_objects, num_features = label(binary_fill_holes(binary_objects))
            dist_field_2d = self.dist_field_3d[0] if self.dist_field_3d is not None else np.zeros_like(mip_img, dtype=float)
            max_geo_dist = self.val(self.lbl_max_geo)
            
            filtered_count = 0
            for i in range(1, num_features + 1):
                if np.sum(labeled_objects == i) < 5: continue
                y_loc, x_loc = [int(v) for v in maximum_position(mip_img, labeled_objects, i)]
                if dist_field_2d[y_loc, x_loc] > max_geo_dist:
                    filtered_count += 1
                    continue
                opt_z, opt_y, opt_x = self.find_optimal_xyz(x_loc, y_loc, search_radius=int(self.val(self.lbl_snap)), start_z=None)
                
                idx = self.target_counter; status = 'new'
                label_text = f"[{idx}] Z:{opt_z+1} Y:{opt_y} X:{opt_x} ({status})"
                self.saved_targets.append({'idx': idx, 'label': label_text, 'z': opt_z, 'y': opt_y, 'x': opt_x,
                                           'click_x': x_loc, 'click_y': y_loc, 'target_type': 'spine', 'status': status, 'is_modified': True})
                self.target_counter += 1
            
            self.update_list_ui()
            if not silent: self.log(f"✅ Extracted {len(self.saved_targets)} seeds (Filtered out {filtered_count} objects > {max_geo_dist} µm away)!")
        if not silent: self.refresh_display()

    def load_csv_seeds(self, seed_type):
        if self.raw_stack is None: return
        dir_name = os.path.dirname(self.files[self.current_idx])
        base_name = os.path.splitext(os.path.basename(self.files[self.current_idx]))[0]
        if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        
        suffix = "seeds_2-respan" if seed_type == 'seed2' else "seeds_3-respan"
        matches = glob(os.path.join(dir_name, f"{base_name}*{suffix}*.csv")) 
        
        if not matches:
            self.log(f"⚠️ No CSV file found matching '*{suffix}*.csv' in the folder.", clear=True)
            return
            
        csv_file = matches[0]
        self.log(f"✅ Found CSV file: {os.path.basename(csv_file)}", clear=True)
        try:
            df = pd.read_csv(csv_file)
            x_col = next((c for c in df.columns if c.lower() in ['x', 'corrected_x', 'centroid_x']), None)
            y_col = next((c for c in df.columns if c.lower() in ['y', 'corrected_y', 'centroid_y']), None)
            if not x_col or not y_col:
                self.log("❌ Could not find an 'x' and 'y' column in the loaded CSV.")
                return
                
            self.saved_targets = []
            if self.input_custom_id.text().strip().isdigit():
                self.target_counter = int(self.input_custom_id.text().strip())
                self.input_custom_id.setText('')
            elif not self.is_correction_mode: 
                self.target_counter = 1
                
            loaded_count = 0
            for _, row in df.iterrows():
                x_loc, y_loc = int(round(row[x_col])), int(round(row[y_col]))
                if not (0 <= y_loc < self.raw_stack.shape[1] and 0 <= x_loc < self.raw_stack.shape[2]): continue
                    
                opt_z, opt_y, opt_x = self.find_optimal_xyz(x_loc, y_loc, search_radius=int(self.val(self.lbl_snap)), start_z=None)
                idx = self.target_counter; status = 'new'
                label_text = f"[{idx}] Z:{opt_z+1} Y:{opt_y} X:{opt_x} ({status})"
                self.saved_targets.append({'idx': idx, 'label': label_text, 'z': opt_z, 'y': opt_y, 'x': opt_x,
                                           'click_x': x_loc, 'click_y': y_loc, 'target_type': 'spine', 'status': status, 'is_modified': True})
                self.target_counter += 1
                loaded_count += 1
                
            self.update_list_ui()
            self.log(f"✅ Loaded and snapped {loaded_count} targets from {os.path.basename(csv_file)}!")
        except Exception as e:
            self.log(f"❌ Error loading CSV: {str(e)}")
        self.refresh_display()

    # --- BATCH PROCESS ---
    def on_analyze_all(self, silent=False):
        if not self.saved_targets or self.raw_stack is None:
            if not silent: self.log("⚠️ Queue is empty. Save targets first.", clear=True)
            return
            
        self.on_save_barrier()
        combined_mask = self.mask.copy()
        results_list = []
        total_barrier = self.get_effective_barrier()
        current_smoothed_stack = np.where(total_barrier, 0, self.base_smoothed_stack)
        
        c_barrier, c_tol, c_zsearch = self.val(self.lbl_barrier), self.val(self.lbl_tol), int(self.val(self.lbl_zsearch))
        
        if not silent: self.log("⚙️ Batch processing targets using 2.5D Dilated Envelope Extrusion...")
        
        for target in self.saved_targets:
            idx = target['idx']
            t_type = 'filopodia' if target.get('is_filopodia', False) else target.get('target_type', 'spine')
            
            if not target.get('is_modified', True) and self.loaded_df is not None:
                row = self.loaded_df[self.loaded_df['Target_ID'] == idx].copy()
                if not row.empty:
                    row_dict = row.iloc[0].to_dict()
                    row_dict['Status'] = target.get('status', row_dict.get('Status', 'static'))
                    row_dict['Classification'] = 'Filopodia' if t_type == 'filopodia' else 'Suboptimal Measures' if t_type == 'suboptimal' else 'Spine'
                    results_list.append(row_dict)
                continue
            
            z, y, x = target['z'], target['y'], target['x']
            orig_x, orig_y = target['click_x'], target['click_y']
            target_status = target.get('status', 'new')
            
            geo_dist = float(self.dist_field_3d[z, y, x]) if self.dist_field_3d is not None else 0.0
            if np.isinf(geo_dist): geo_dist = 0.0
            
            if t_type == 'filopodia':
                results_list.append({'Target_ID': idx, 'Classification': 'Filopodia', 'Status': target_status, 'Z_Slice': z + 1, 
                                     'Original_Y': orig_y, 'Original_X': orig_x, 'Corrected_Y': y, 'Corrected_X': x, 
                                     'Local_Dendrite_Surface_Max': 0, 'Local_Dendrite_Surface_IntDen': 0.0, 'Area_Opt_Z_um2': 0.0,
                                     'Geodesic_Distance_um': geo_dist, 'Vol_voxels': 0, 'Vol_um3': 0.0, 'Max_Intensity': int(self.raw_stack[z,y,x]),
                                     'Sum_Intensity': 0.0, 'Integrated_Density': 0.0, 'Z_Slices_Count': 0, 
                                     'Avg_Initial_Dendrite_Intensity': self.avg_initial_dendrite_intensity, 'Barrier_um': c_barrier,
                                     'Tolerance': c_tol, 'Z_Search_Range': c_zsearch, 'Dendrite_Length_um': self.dendrite_length_um})
                continue
                
            pad_xy, pad_z = 30, c_zsearch
            z_min_loc = int(max(0, z - pad_z))
            z_max_loc = int(min(self.raw_stack.shape[0], z + pad_z + 1))
            y_min_loc = int(max(0, y - pad_xy))
            y_max_loc = int(min(self.raw_stack.shape[1], y + pad_xy + 1))
            x_min_loc = int(max(0, x - pad_xy))
            x_max_loc = int(min(self.raw_stack.shape[2], x + pad_xy + 1))
            
            lz, ly, lx = int(z - z_min_loc), int(y - y_min_loc), int(x - x_min_loc)

            seed_val = self.base_smoothed_stack[z, y, x] if (self.erased_barrier_2d is not None and self.erased_barrier_2d[y, x]) else current_smoothed_stack[z, y, x]
            if seed_val == 0:
                if not silent: self.log(f"❌ Target [{idx}] skipped: Seed is inside the pink dendritic barrier.")
                continue
                
            sub_stack = current_smoothed_stack[z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc]
            lower_bound = max(seed_val * (1.0 - c_tol), 1e-6)
            
            slice_2d = sub_stack[lz]
            binary_2d = slice_2d >= lower_bound
            if self.chk_hessian.isChecked():
                blob_mask_2d = self.get_2d_hessian_blob_mask(np.expand_dims(slice_2d, 0), self.val(self.lbl_strict), sig=self.val(self.lbl_sigma))[0]
                blob_mask_2d[ly, lx] = True
                binary_2d = binary_2d & blob_mask_2d
                
            labeled_2d, _ = label(binary_2d)
            if labeled_2d[ly, lx] == 0:
                if not silent: self.log(f"❌ Target [{idx}] skipped: 2D seed slice failed threshold or topology.")
                continue
                
            envelope_3d = np.broadcast_to(binary_dilation(binary_fill_holes(labeled_2d == labeled_2d[ly, lx]), iterations=3), sub_stack.shape)
            
            local_binary = sub_stack >= lower_bound
            if self.chk_hessian.isChecked():
                blob_mask_3d = self.get_2d_hessian_blob_mask(sub_stack, self.val(self.lbl_strict), sig=self.val(self.lbl_sigma))
                blob_mask_3d[lz, ly, lx] = True 
                local_binary = local_binary & blob_mask_3d
                
            local_binary = local_binary & envelope_3d
            labeled_local, _ = label(local_binary)
            if labeled_local[lz, ly, lx] == 0:
                if not silent: self.log(f"❌ Target [{idx}] skipped: 3D region growing failed inside envelope.")
                continue
                
            local_spine_mask = binary_fill_holes(labeled_local == labeled_local[lz, ly, lx])
            global_target_mask = np.zeros_like(combined_mask, dtype=bool)
            global_target_mask[z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc] = local_spine_mask
            
            class_label = 'Suboptimal Measures' if t_type == 'suboptimal' else 'Spine'
            if t_type == 'spine': combined_mask[global_target_mask] = idx
            
            voxels = np.sum(global_target_mask)
            vol = voxels * self.voxel_volume
            max_intensity = int(self.raw_stack[global_target_mask].max())
            int_density = np.sum(self.raw_stack[global_target_mask], dtype=np.float64)
            area_opt_z_um2 = np.sum(global_target_mask[z, :, :]) * (self.dx * self.dy)
            
            local_barrier = total_barrier[z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc]
            local_raw = self.raw_stack[z_min_loc:z_max_loc, y_min_loc:y_max_loc, x_min_loc:x_max_loc]
            local_barrier_surface = local_barrier & ~binary_erosion(local_barrier)
            
            if np.any(local_barrier_surface):
                local_dend_surf_max = int(local_raw[local_barrier_surface].max())
                local_dend_surf_sum = float(np.sum(local_raw[local_barrier_surface]))
            else: local_dend_surf_max, local_dend_surf_sum = 0, 0.0
            
            results_list.append({'Target_ID': idx, 'Classification': class_label, 'Status': target_status, 'Z_Slice': z + 1, 
                                 'Original_Y': orig_y, 'Original_X': orig_x, 'Corrected_Y': y, 'Corrected_X': x, 
                                 'Local_Dendrite_Surface_Max': local_dend_surf_max, 'Local_Dendrite_Surface_IntDen': local_dend_surf_sum,
                                 'Area_Opt_Z_um2': area_opt_z_um2, 'Geodesic_Distance_um': geo_dist, 'Vol_voxels': voxels, 'Vol_um3': vol,
                                 'Max_Intensity': max_intensity, 'Sum_Intensity': int_density, 'Integrated_Density': int_density,
                                 'Z_Slices_Count': int(np.sum(np.any(global_target_mask, axis=(1, 2)))), 
                                 'Avg_Initial_Dendrite_Intensity': self.avg_initial_dendrite_intensity, 'Barrier_um': c_barrier,
                                 'Tolerance': c_tol, 'Z_Search_Range': c_zsearch, 'Dendrite_Length_um': self.dendrite_length_um})
            
        self.mask = combined_mask
        final_results_df = pd.DataFrame(results_list)
        
        in_path = self.input_folder if self.input_folder else os.path.dirname(self.files[self.current_idx])
        out_dir = os.path.join(in_path, 'output_analysis')
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.files[self.current_idx]))[0]
        if base_name.endswith('.tif'): base_name = os.path.splitext(base_name)[0]
        
        prefix = '_corrected' if self.is_correction_mode else ''
        final_results_df.to_csv(os.path.join(out_dir, base_name + f'{prefix}_spine_results.csv'), index=False)
        imwrite(os.path.join(out_dir, base_name + f'{prefix}_filtered.tif'), current_smoothed_stack)
        imwrite(os.path.join(out_dir, base_name + f'{prefix}_segmentation_mask.tif'), combined_mask.astype(np.uint16))
        
        if not silent:
            self.chk_segment.setChecked(True)
            self.refresh_display()
            
            mip_img_raw = np.max(self.raw_stack, axis=0)
            mip_norm = mip_img_raw.astype(float)
            if mip_norm.max() > mip_norm.min(): mip_norm = (255 * (mip_norm - mip_norm.min()) / (mip_norm.max() - mip_norm.min())).astype(np.uint8)
            else: mip_norm = np.zeros_like(mip_norm, dtype=np.uint8)
            mip_rgb = np.stack([mip_norm, mip_norm, mip_norm], axis=-1)
            
            mip_mask = np.max(combined_mask, axis=0) > 0
            mip_rgb[mip_mask, 0] = 0; mip_rgb[mip_mask, 1] = 255; mip_rgb[mip_mask, 2] = 0
            
            fig_mip, ax_mip = plt.subplots(figsize=(8, 8), dpi=150)
            ax_mip.imshow(mip_rgb); ax_mip.axis('off'); ax_mip.set_ylim(mip_rgb.shape[0], 0)
            
            for _, row in final_results_df.iterrows():
                rx, ry, tid = int(row['Corrected_X']), int(row['Corrected_Y']), int(row['Target_ID'])
                c_class = row['Classification']
                color = 'cyan' if c_class == 'Filopodia' else 'orange' if c_class == 'Suboptimal Measures' else 'yellow'
                ax_mip.plot(rx, ry, '.', color=color, label=str(tid))
                ax_mip.text(rx + 3, ry, str(tid), color=color, fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
                
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig_mip.savefig(os.path.join(out_dir, base_name + '_mip_segmented.png'), bbox_inches='tight', pad_inches=0)
            plt.close(fig_mip)
            
            self.log(f"✅ Batch Analysis Complete! Processed {len(final_results_df)} targets. Saved to {os.path.basename(out_dir)}/")

    def is_valid_image(self, filename):
        fn = filename.lower()
        exclusions = [
            '_dendrite-geo.tif', '_dendrite-geo.tiff',
            '_dendrite-respan.tif', '_dendrite-respan.tiff',
            '_segmentation_mask.tif', '_segmentation_mask.tiff',
            '_filtered.tif', '_filtered.tiff',
            '_custom_barrier_2d.tif', '_custom_barrier_2d.tiff',
            '.swc', '.csv', '.png'
        ]
        for exc in exclusions:
            if fn.endswith(exc):
                return False
        return True

    def on_load_folder(self):
        if not os.path.isdir(self.input_folder):
            self.log("❌ Invalid input folder path.", clear=True)
            return
            
        out_dir = os.path.join(self.input_folder, 'output_analysis')
        os.makedirs(out_dir, exist_ok=True)
            
        all_files = sorted(glob(os.path.join(self.input_folder, "*.tif")) + glob(os.path.join(self.input_folder, "*.tiff")))
        remaining_files = []
        for f in all_files:
            fn = os.path.basename(f)
            if not self.is_valid_image(fn): continue
                
            bn = os.path.splitext(fn)[0]
            if bn.endswith('.tif'): bn = os.path.splitext(bn)[0]
            if not os.path.exists(os.path.join(out_dir, bn + '_spine_results.csv')) and not os.path.exists(os.path.join(out_dir, bn + '_corrected_spine_results.csv')): 
                remaining_files.append(f)
                
        if not remaining_files:
            self.log("🎉 All images in this folder have already been analyzed!", clear=True)
            return
            
        self.files = remaining_files
        self.current_idx = 0
        self.is_correction_mode = False
        self.load_image_at_index(0)

    def on_load_analyzed_folder(self):
        if not os.path.isdir(self.input_folder):
            self.log("❌ Invalid input folder path.", clear=True)
            return
            
        out_dir = os.path.join(self.input_folder, 'output_analysis')
        all_files = sorted(glob(os.path.join(self.input_folder, "*.tif")) + glob(os.path.join(self.input_folder, "*.tiff")))
        analyzed_files = []
        for f in all_files:
            fn = os.path.basename(f)
            if not self.is_valid_image(fn): continue
                
            bn = os.path.splitext(fn)[0]
            if bn.endswith('.tif'): bn = os.path.splitext(bn)[0]
            if os.path.exists(os.path.join(out_dir, bn + '_spine_results.csv')) or os.path.exists(os.path.join(out_dir, bn + '_corrected_spine_results.csv')):
                analyzed_files.append(f)
                
        if not analyzed_files:
            self.log("⚠️ No analyzed files found in the output directory.", clear=True)
            return
            
        self.files = analyzed_files
        self.current_idx = 0
        self.is_correction_mode = True
        self.load_image_at_index(0)

    def on_batch_unattended(self):
        self.on_load_folder()
        if not self.files: return
        self.log(f"🚀 Starting Unattended Batch Process for {len(self.files)} files...", clear=True)
        for idx in range(len(self.files)):
            self.load_image_at_index(idx)
            self.current_idx = idx
            
            if np.isinf(self.dist_field_3d).all():
                self.log(f"⚠️ No mask found for {os.path.basename(self.files[idx])}. Auto-generating barrier...")
                self.on_auto_generate_barrier()
                
            self.auto_generate_seeds(silent=True)
            self.on_analyze_all(silent=True)
        self.log("🎉 Unattended batch completed successfully!")

    def on_prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image_at_index(self.current_idx)
        else:
            self.log("⚠️ Already at the first image in the queue!", clear=True)

    def on_next_image(self):
        if self.current_idx < len(self.files) - 1:
            self.current_idx += 1
            self.load_image_at_index(self.current_idx)
        else:
            self.log("🎉 Completed all files in the queue!", clear=True)

if __name__ == '__main__':
    if os.name == 'nt':
        myappid = 'ufscripps.intspine.analyzer.v1'
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except AttributeError:
            pass

    app = QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(__file__), 'intspine_logo.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    window = SpineAnalyzerApp()
    window.show()
    sys.exit(app.exec())