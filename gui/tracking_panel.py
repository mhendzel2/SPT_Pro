"""
Tracking panel module for SPT Analysis GUI.

This module provides the tracking panel for the SPT Analysis GUI, with controls for
particle detection, linking, tracking operations, and compartment analysis.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, 
    QCheckBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget, 
    QTextEdit, QFileDialog, QMessageBox, QProgressBar, QListWidget, QSlider,
    QColorDialog, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import os
import logging

from ..tracking import detector, linker, tracker, multi_channel
from ..analysis import compartment_analyzer
from ..utils import io, processing, visualization
from ..project import management

logger = logging.getLogger(__name__)


class TrackingPanel(QWidget):
    """
    Panel for configuring and running particle tracking operations.
    
    This panel provides controls for:
    - Loading image data
    - Configuring detection parameters
    - Setting linking algorithms
    - Running tracking operations
    - Compartment segmentation and analysis
    - Visualizing tracking results
    """
    
    # Signal emitted when tracking is complete
    tracking_complete = pyqtSignal(object)
    
    # Signal emitted when compartment analysis is complete
    compartment_analysis_complete = pyqtSignal(object)
    
    def __init__(self, parent=None, project_manager=None):
        """
        Initialize the tracking panel.
        
        Parameters
        ----------
        parent : QWidget, optional
            The parent widget
        project_manager : ProjectManager, optional
            Project manager instance for handling project operations
        """
        super().__init__(parent)
        self.project_manager = project_manager or management.ProjectManager()
        
        # Initialize tracking data
        self.image_data = None
        self.detection_results = None
        self.tracking_results = None
        self.compartment_results = None
        self.compartment_definitions = []
        # Initialize default compartment color
        self.compartment_color = QColor(0, 255, 0, 128)
        
        # Setup UI components
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface components."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # File operations section
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Images")
        self.load_button.clicked.connect(self._on_load_images)
        
        self.save_config_button = QPushButton("Save Configuration")
        self.save_config_button.clicked.connect(self._on_save_config)
        
        self.load_config_button = QPushButton("Load Configuration")
        self.load_config_button.clicked.connect(self._on_load_config)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._on_export_results)
        self.export_button.setEnabled(False)
        
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.save_config_button)
        file_layout.addWidget(self.load_config_button)
        file_layout.addWidget(self.export_button)
        file_layout.addStretch()
        
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # Status indicator
        status_layout = QHBoxLayout()
        self.status_label = QLabel("No data loaded")
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)
        
        # Configuration tabs
        config_tabs = QTabWidget()
        
        # Detection tab
        detection_tab = QWidget()
        detection_layout = QVBoxLayout()
        
        # Detection method
        method_group = QGroupBox("Detection Method")
        method_form = QFormLayout()
        
        self.detection_method = QComboBox()
        self.detection_method.addItems([
            "gaussian", 
            "laplacian", 
            "doh", 
            "log", 
            "unet",  # Advanced deep learning method
            "wavelet"  # Wavelet-based detection
        ])
        self.detection_method.currentTextChanged.connect(self._on_detection_method_changed)
        method_form.addRow("Method:", self.detection_method)
        
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.5)
        method_form.addRow("Threshold:", self.threshold)
        
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 100)
        self.min_distance.setValue(5)
        method_form.addRow("Min Distance (px):", self.min_distance)
        
        self.diameter = QSpinBox()
        self.diameter.setRange(3, 51)
        self.diameter.setSingleStep(2)
        self.diameter.setValue(7)
        method_form.addRow("Diameter (px):", self.diameter)
        
        self.subpixel = QCheckBox("Subpixel localization")
        self.subpixel.setChecked(True)
        method_form.addRow(self.subpixel)
        
        # Advanced detection parameters (initially hidden)
        self.model_path_label = QLabel("Model Path:")
        self.model_path_input = QPushButton("Select Model...")
        self.model_path_input.clicked.connect(self._on_select_model)
        self.model_path = ""
        
        self.use_gpu = QCheckBox("Use GPU if available")
        self.use_gpu.setChecked(True)
        
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.1, 0.99)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setValue(0.5)
        
        # Hide advanced params initially
        self.model_path_label.setVisible(False)
        self.model_path_input.setVisible(False)
        self.use_gpu.setVisible(False)
        self.confidence_threshold.setVisible(False)
        
        method_form.addRow(self.model_path_label, self.model_path_input)
        method_form.addRow("Confidence:", self.confidence_threshold)
        method_form.addRow(self.use_gpu)
        
        method_group.setLayout(method_form)
        detection_layout.addWidget(method_group)
        
        # Test detection button
        self.test_detection_button = QPushButton("Test Detection")
        self.test_detection_button.clicked.connect(self._on_test_detection)
        self.test_detection_button.setEnabled(False)
        detection_layout.addWidget(self.test_detection_button)
        
        detection_layout.addStretch()
        detection_tab.setLayout(detection_layout)
        config_tabs.addTab(detection_tab, "Detection")
        
        # Linking tab
        linking_tab = QWidget()
        linking_layout = QVBoxLayout()
        
        # Linking method
        linking_group = QGroupBox("Linking Parameters")
        linking_form = QFormLayout()
        
        self.linking_method = QComboBox()
        self.linking_method.addItems(["nearest_neighbor", "hungarian", "trackpy"])
        linking_form.addRow("Method:", self.linking_method)
        
        self.max_distance = QDoubleSpinBox()
        self.max_distance.setRange(1.0, 100.0)
        self.max_distance.setSingleStep(0.5)
        self.max_distance.setValue(10.0)
        linking_form.addRow("Max Distance (px):", self.max_distance)
        
        self.max_gap_frames = QSpinBox()
        self.max_gap_frames.setRange(0, 10)
        self.max_gap_frames.setValue(1)
        linking_form.addRow("Max Gap Frames:", self.max_gap_frames)
        
        self.memory = QSpinBox()
        self.memory.setRange(0, 10)
        self.memory.setValue(0)
        linking_form.addRow("Memory:", self.memory)
        
        linking_group.setLayout(linking_form)
        linking_layout.addWidget(linking_group)
        linking_layout.addStretch()
        linking_tab.setLayout(linking_layout)
        config_tabs.addTab(linking_tab, "Linking")
        
        # Tracker tab
        tracker_tab = QWidget()
        tracker_layout = QVBoxLayout()
        
        # Tracker settings
        tracker_group = QGroupBox("Tracker Parameters")
        tracker_form = QFormLayout()
        
        self.min_track_length = QSpinBox()
        self.min_track_length.setRange(1, 100)
        self.min_track_length.setValue(3)
        tracker_form.addRow("Min Track Length:", self.min_track_length)
        
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.01, 10.0)
        self.pixel_size.setSingleStep(0.01)
        self.pixel_size.setValue(0.1)
        self.pixel_size.setSuffix(" µm")
        tracker_form.addRow("Pixel Size:", self.pixel_size)
        
        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setRange(0.001, 60.0)
        self.frame_interval.setSingleStep(0.001)
        self.frame_interval.setValue(0.1)
        self.frame_interval.setSuffix(" s")
        tracker_form.addRow("Frame Interval:", self.frame_interval)
        
        tracker_group.setLayout(tracker_form)
        tracker_layout.addWidget(tracker_group)
        
        # Multi-channel settings
        multi_channel_group = QGroupBox("Multi-Channel Settings")
        multi_channel_layout = QVBoxLayout()
        
        self.enable_multi_channel = QCheckBox("Enable Multi-Channel")
        self.enable_multi_channel.setChecked(False)
        self.enable_multi_channel.toggled.connect(self._toggle_multi_channel)
        multi_channel_layout.addWidget(self.enable_multi_channel)
        
        multi_channel_form = QFormLayout()
        
        self.channel_count = QSpinBox()
        self.channel_count.setRange(1, 10)
        self.channel_count.setValue(1)
        self.channel_count.setEnabled(False)
        multi_channel_form.addRow("Channel Count:", self.channel_count)
        
        self.reference_channel = QSpinBox()
        self.reference_channel.setRange(0, 9)
        self.reference_channel.setValue(0)
        self.reference_channel.setEnabled(False)
        multi_channel_form.addRow("Reference Channel:", self.reference_channel)
        
        multi_channel_layout.addLayout(multi_channel_form)
        multi_channel_group.setLayout(multi_channel_layout)
        tracker_layout.addWidget(multi_channel_group)
        
        tracker_layout.addStretch()
        tracker_tab.setLayout(tracker_layout)
        config_tabs.addTab(tracker_tab, "Tracker")
        
        # Compartment Analysis tab
        compartment_tab = QWidget()
        compartment_layout = QVBoxLayout()
        
        # Enable compartment analysis
        self.enable_compartment = QCheckBox("Enable Compartment Analysis")
        self.enable_compartment.setChecked(False)
        self.enable_compartment.toggled.connect(self._toggle_compartment)
        compartment_layout.addWidget(self.enable_compartment)
        
        # Compartment settings
        compartment_group = QGroupBox("Compartment Definitions")
        compartment_group.setEnabled(False)
        self.compartment_group = compartment_group
        compartment_group_layout = QVBoxLayout()
        
        # List of defined compartments
        self.compartment_list = QListWidget()
        self.compartment_list.itemSelectionChanged.connect(self._on_compartment_selection_changed)
        compartment_group_layout.addWidget(self.compartment_list)
        
        # Compartment buttons
        compartment_buttons = QHBoxLayout()
        
        self.add_compartment_button = QPushButton("Add")
        self.add_compartment_button.clicked.connect(self._on_add_compartment)
        
        self.remove_compartment_button = QPushButton("Remove")
        self.remove_compartment_button.clicked.connect(self._on_remove_compartment)
        self.remove_compartment_button.setEnabled(False)
        
        compartment_buttons.addWidget(self.add_compartment_button)
        compartment_buttons.addWidget(self.remove_compartment_button)
        compartment_group_layout.addLayout(compartment_buttons)
        
        # Compartment properties
        compartment_props = QFormLayout()
        
        self.compartment_name = QComboBox()
        self.compartment_name.addItems([
            "Nucleus", 
            "Cytoplasm", 
            "Membrane", 
            "Vesicle", 
            "Mitochondria", 
            "Endosome", 
            "Lysosome", 
            "Custom"
        ])
        compartment_props.addRow("Type:", self.compartment_name)
        
        self.compartment_channel = QSpinBox()
        self.compartment_channel.setRange(0, 9)
        self.compartment_channel.setValue(1)  # Typically channel 0 is for particles
        compartment_props.addRow("Channel:", self.compartment_channel)
        
        self.compartment_threshold = QDoubleSpinBox()
        self.compartment_threshold.setRange(0.0, 1.0)
        self.compartment_threshold.setSingleStep(0.05)
        self.compartment_threshold.setValue(0.5)
        compartment_props.addRow("Threshold:", self.compartment_threshold)
        
        self.compartment_min_size = QSpinBox()
        self.compartment_min_size.setRange(10, 10000)
        self.compartment_min_size.setValue(100)
        compartment_props.addRow("Min Size (px²):", self.compartment_min_size)
        
        self.compartment_color_button = QPushButton("Set Color")
        self.compartment_color_button.clicked.connect(self._on_select_compartment_color)
        compartment_props.addRow("Color:", self.compartment_color_button)
        
        compartment_group_layout.addLayout(compartment_props)
        
        compartment_group.setLayout(compartment_group_layout)
        compartment_layout.addWidget(compartment_group)
        
        # Segmentation method and parameters
        segmentation_group = QGroupBox("Segmentation Method")
        segmentation_group.setEnabled(False)
        self.segmentation_group = segmentation_group
        segmentation_layout = QFormLayout()
        
        self.segmentation_method = QComboBox()
        self.segmentation_method.addItems([
            "Threshold",
            "Watershed",
            "Active Contour",
            "Deep Learning"
        ])
        self.segmentation_method.currentTextChanged.connect(self._on_segmentation_method_changed)
        segmentation_layout.addRow("Method:", self.segmentation_method)
        
        self.apply_preprocessing = QCheckBox("Apply Preprocessing")
        self.apply_preprocessing.setChecked(True)
        segmentation_layout.addRow(self.apply_preprocessing)
        
        self.fill_holes = QCheckBox("Fill Holes")
        self.fill_holes.setChecked(True)
        segmentation_layout.addRow(self.fill_holes)
        
        self.smooth_boundaries = QCheckBox("Smooth Boundaries")
        self.smooth_boundaries.setChecked(True)
        segmentation_layout.addRow(self.smooth_boundaries)
        
        segmentation_group.setLayout(segmentation_layout)
        compartment_layout.addWidget(segmentation_group)
        
        # Test segmentation button
        self.test_segmentation_button = QPushButton("Test Segmentation")
        self.test_segmentation_button.clicked.connect(self._on_test_segmentation)
        self.test_segmentation_button.setEnabled(False)
        compartment_layout.addWidget(self.test_segmentation_button)
        
        compartment_layout.addStretch()
        compartment_tab.setLayout(compartment_layout)
        config_tabs.addTab(compartment_tab, "Compartments")
        
        main_layout.addWidget(config_tabs)
        
        # Run tracking button
        self.run_tracking_button = QPushButton("Run Tracking")
        self.run_tracking_button.clicked.connect(self._on_run_tracking)
        self.run_tracking_button.setEnabled(False)
        main_layout.addWidget(self.run_tracking_button)
        
        # Results visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        # Create a matplotlib figure for visualization
        self.figure = plt.figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Visualization buttons
        viz_buttons_layout = QHBoxLayout()
        
        self.show_tracks_button = QPushButton("Show Tracks")
        self.show_tracks_button.clicked.connect(self._on_show_tracks)
        self.show_tracks_button.setEnabled(False)
        
        self.show_histogram_button = QPushButton("Show Histogram")
        self.show_histogram_button.clicked.connect(self._on_show_histogram)
        self.show_histogram_button.setEnabled(False)
        
        self.show_compartments_button = QPushButton("Show Compartments")
        self.show_compartments_button.clicked.connect(self._on_show_compartments)
        self.show_compartments_button.setEnabled(False)
        
        self.save_figure_button = QPushButton("Save Figure")
        self.save_figure_button.clicked.connect(self._on_save_figure)
        self.save_figure_button.setEnabled(False)
        
        viz_buttons_layout.addWidget(self.show_tracks_button)
        viz_buttons_layout.addWidget(self.show_histogram_button)
        viz_buttons_layout.addWidget(self.show_compartments_button)
        viz_buttons_layout.addWidget(self.save_figure_button)
        viz_buttons_layout.addStretch()
        
        viz_layout.addLayout(viz_buttons_layout)
        viz_group.setLayout(viz_layout)
        main_layout.addWidget(viz_group)
    
    def _on_detection_method_changed(self, method):
        """Handle change of detection method."""
        # Show/hide advanced parameters based on method
        is_advanced = method in ["unet", "wavelet"]
        
        self.model_path_label.setVisible(is_advanced and method == "unet")
        self.model_path_input.setVisible(is_advanced and method == "unet")
        self.use_gpu.setVisible(is_advanced)
        self.confidence_threshold.setVisible(is_advanced)
        
        # Adjust threshold range based on method
        if method in ["gaussian", "laplacian"]:
            self.threshold.setRange(0.01, 1.0)
            self.threshold.setValue(0.5)
        elif method in ["doh", "log"]:
            self.threshold.setRange(0.001, 0.1)
            self.threshold.setValue(0.01)
        elif method == "unet":
            self.threshold.setRange(0.5, 0.99)
            self.threshold.setValue(0.7)
        elif method == "wavelet":
            self.threshold.setRange(1.0, 10.0)
            self.threshold.setValue(3.0)
    
    def _on_select_model(self):
        """Select a model file for deep learning detection."""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pt *.pth);;All Files (*.*)"
        )
        
        if model_path:
            self.model_path = model_path
            self.model_path_input.setToolTip(model_path)
            # Show just the filename on the button
            self.model_path_input.setText(os.path.basename(model_path))
    
    def _toggle_multi_channel(self, enabled):
        """Enable or disable multi-channel settings based on checkbox state."""
        self.channel_count.setEnabled(enabled)
        self.reference_channel.setEnabled(enabled)
    
    def _toggle_compartment(self, enabled):
        """Enable or disable compartment analysis settings."""
        self.compartment_group.setEnabled(enabled)
        self.segmentation_group.setEnabled(enabled)
        self.test_segmentation_button.setEnabled(enabled and self.image_data is not None)
        
        # Enable compartment visualization if we have results
        if self.compartment_results is not None:
            self.show_compartments_button.setEnabled(enabled)
    
    def _on_segmentation_method_changed(self, method):
        """Handle change of segmentation method."""
        # Future: Add specific controls for each segmentation method
        pass
    
    def _on_compartment_selection_changed(self):
        """Handle selection change in compartment list."""
        self.remove_compartment_button.setEnabled(len(self.compartment_list.selectedItems()) > 0)
        
        # Load selected compartment settings
        selected_items = self.compartment_list.selectedItems()
        if selected_items:
            compartment_name = selected_items[0].text()
            for comp in self.compartment_definitions:
                if comp.name == compartment_name:
                    # Update UI with compartment properties
                    index = self.compartment_name.findText(comp.name)
                    if index >= 0:
                        self.compartment_name.setCurrentIndex(index)
                    else:
                        # Use "Custom" if not found
                        index = self.compartment_name.findText("Custom")
                        self.compartment_name.setCurrentIndex(index)
                    
                    self.compartment_channel.setValue(comp.channel)
                    self.compartment_threshold.setValue(comp.threshold)
                    self.compartment_min_size.setValue(comp.min_size)
                    self.compartment_color = comp.color
                    # Update color button style
                    self.compartment_color_button.setStyleSheet(
                        f"background-color: rgba({comp.color.red()}, {comp.color.green()}, "
                        f"{comp.color.blue()}, {comp.color.alpha()});"
                    )
                    break
    
    def _on_add_compartment(self):
        """Add a new compartment definition."""
        # Get properties from UI
        name = self.compartment_name.currentText()
        if name == "Custom":
            # Ask for custom name
            name, ok = QInputDialog.getText(self, "Custom Compartment", "Enter name:")
            if not ok or not name:
                return
        
        # Check if name already exists
        for i in range(self.compartment_list.count()):
            if self.compartment_list.item(i).text() == name:
                QMessageBox.warning(self, "Duplicate Name", 
                                     "A compartment with this name already exists.")
                return
        
        # Create compartment definition
        compartment = compartment_analyzer.CompartmentDefinition(
            name=name,
            channel=self.compartment_channel.value(),
            threshold=self.compartment_threshold.value(),
            min_size=self.compartment_min_size.value(),
            color=self.compartment_color
        )
        
        # Add to list
        self.compartment_definitions.append(compartment)
        self.compartment_list.addItem(name)
        
        # Select the new item
        for i in range(self.compartment_list.count()):
            if self.compartment_list.item(i).text() == name:
                self.compartment_list.setCurrentRow(i)
                break
    
    def _on_remove_compartment(self):
        """Remove the selected compartment definition."""
        selected_items = self.compartment_list.selectedItems()
        if not selected_items:
            return
        
        compartment_name = selected_items[0].text()
        
        # Remove from list
        for i, comp in enumerate(self.compartment_definitions):
            if comp.name == compartment_name:
                self.compartment_definitions.pop(i)
                break
        
        # Remove from list widget
        for i in range(self.compartment_list.count()):
            if self.compartment_list.item(i).text() == compartment_name:
                self.compartment_list.takeItem(i)
                break
        
        # Update button state
        self.remove_compartment_button.setEnabled(False)
    
    def _on_select_compartment_color(self):
        """Open color dialog to select compartment color."""
        color = QColorDialog.getColor(
            self.compartment_color, 
            self, 
            "Select Compartment Color",
            QColorDialog.ShowAlphaChannel
        )
        
        if color.isValid():
            self.compartment_color = color
            # Update color button style
            self.compartment_color_button.setStyleSheet(
                f"background-color: rgba({color.red()}, {color.green()}, "
                f"{color.blue()}, {color.alpha()});"
            )
    
    def _on_load_images(self):
        """Load image data for tracking."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image Data", 
            "", 
            "Image Files (*.tif *.tiff);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Use the IO module to load the image data
            self.image_data = io.load_image(file_path)
            
            # Update status and enable buttons
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            self.test_detection_button.setEnabled(True)
            self.run_tracking_button.setEnabled(True)
            
            # Enable test segmentation if compartment analysis is enabled
            if self.enable_compartment.isChecked():
                self.test_segmentation_button.setEnabled(True)
            
            # Display first frame in the visualization area
            self._display_frame(0)
            
            logger.info(f"Loaded image data from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image data: {str(e)}")
            logger.error(f"Error loading image data: {str(e)}")
    
    def _on_save_config(self):
        """Save current tracking configuration to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Configuration", 
            "", 
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Collect configuration from UI components
            config_dict = {
                'detector': {
                    'method': self.detection_method.currentText(),
                    'threshold': self.threshold.value(),
                    'min_distance': self.min_distance.value(),
                    'diameter': self.diameter.value(),
                    'subpixel': self.subpixel.isChecked(),
                    'model_path': self.model_path,
                    'use_gpu': self.use_gpu.isChecked(),
                    'confidence_threshold': self.confidence_threshold.value()
                },
                'linker': {
                    'method': self.linking_method.currentText(),
                    'max_distance': self.max_distance.value(),
                    'max_gap_frames': self.max_gap_frames.value(),
                    'memory': self.memory.value()
                },
                'tracker': {
                    'min_track_length': self.min_track_length.value(),
                    'pixel_size': self.pixel_size.value(),
                    'frame_interval': self.frame_interval.value()
                },
                'multi_channel': {
                    'enabled': self.enable_multi_channel.isChecked(),
                    'channel_count': self.channel_count.value(),
                    'reference_channel': self.reference_channel.value()
                },
                'compartment_analysis': {
                    'enabled': self.enable_compartment.isChecked(),
                    'segmentation_method': self.segmentation_method.currentText(),
                    'apply_preprocessing': self.apply_preprocessing.isChecked(),
                    'fill_holes': self.fill_holes.isChecked(),
                    'smooth_boundaries': self.smooth_boundaries.isChecked(),
                    'compartments': [comp.to_dict() for comp in self.compartment_definitions]
                }
            }
            
            # Use the project manager to save the configuration
            self.project_manager.save_config(config_dict, file_path)
            
            QMessageBox.information(self, "Success", "Configuration saved successfully")
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")
            logger.error(f"Error saving configuration: {str(e)}")
    
    def _on_load_config(self):
        """Load tracking configuration from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Configuration", 
            "", 
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Use the project manager to load the configuration
            config_dict = self.project_manager.load_config(file_path)
            
            # Update UI components with loaded values
            if 'detector' in config_dict:
                det = config_dict['detector']
                if 'method' in det:
                    index = self.detection_method.findText(det['method'])
                    if index >= 0:
                        self.detection_method.setCurrentIndex(index)
                if 'threshold' in det:
                    self.threshold.setValue(det['threshold'])
                if 'min_distance' in det:
                    self.min_distance.setValue(det['min_distance'])
                if 'diameter' in det:
                    self.diameter.setValue(det['diameter'])
                if 'subpixel' in det:
                    self.subpixel.setChecked(det['subpixel'])
                if 'model_path' in det:
                    self.model_path = det['model_path']
                    if self.model_path:
                        self.model_path_input.setToolTip(self.model_path)
                        self.model_path_input.setText(os.path.basename(self.model_path))
                if 'use_gpu' in det:
                    self.use_gpu.setChecked(det['use_gpu'])
                if 'confidence_threshold' in det:
                    self.confidence_threshold.setValue(det['confidence_threshold'])
            
            if 'linker' in config_dict:
                link = config_dict['linker']
                if 'method' in link:
                    index = self.linking_method.findText(link['method'])
                    if index >= 0:
                        self.linking_method.setCurrentIndex(index)
                if 'max_distance' in link:
                    self.max_distance.setValue(link['max_distance'])
                if 'max_gap_frames' in link:
                    self.max_gap_frames.setValue(link['max_gap_frames'])
                if 'memory' in link:
                    self.memory.setValue(link['memory'])
            
            if 'tracker' in config_dict:
                track = config_dict['tracker']
                if 'min_track_length' in track:
                    self.min_track_length.setValue(track['min_track_length'])
                if 'pixel_size' in track:
                    self.pixel_size.setValue(track['pixel_size'])
                if 'frame_interval' in track:
                    self.frame_interval.setValue(track['frame_interval'])
            
            if 'multi_channel' in config_dict:
                mc = config_dict['multi_channel']
                if 'enabled' in mc:
                    self.enable_multi_channel.setChecked(mc['enabled'])
                if 'channel_count' in mc:
                    self.channel_count.setValue(mc['channel_count'])
                if 'reference_channel' in mc:
                    self.reference_channel.setValue(mc['reference_channel'])
            
            if 'compartment_analysis' in config_dict:
                comp = config_dict['compartment_analysis']
                if 'enabled' in comp:
                    self.enable_compartment.setChecked(comp['enabled'])
                if 'segmentation_method' in comp:
                    index = self.segmentation_method.findText(comp['segmentation_method'])
                    if index >= 0:
                        self.segmentation_method.setCurrentIndex(index)
                if 'apply_preprocessing' in comp:
                    self.apply_preprocessing.setChecked(comp['apply_preprocessing'])
                if 'fill_holes' in comp:
                    self.fill_holes.setChecked(comp['fill_holes'])
                if 'smooth_boundaries' in comp:
                    self.smooth_boundaries.setChecked(comp['smooth_boundaries'])
                
                # Load compartment definitions
                if 'compartments' in comp:
                    # Clear existing definitions
                    self.compartment_definitions = []
                    self.compartment_list.clear()
                    
                    for comp_dict in comp['compartments']:
                        try:
                            compartment = compartment_analyzer.CompartmentDefinition.from_dict(comp_dict)
                            self.compartment_definitions.append(compartment)
                            self.compartment_list.addItem(compartment.name)
                        except Exception as e:
                            logger.warning(f"Failed to load compartment: {str(e)}")
            
            QMessageBox.information(self, "Success", "Configuration loaded successfully")
            logger.info(f"Configuration loaded from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _on_export_results(self):
        """Export tracking and analysis results to a file."""
        if self.tracking_results is None:
            QMessageBox.warning(self, "Warning", "No tracking results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Results", 
            "", 
            "CSV Files (*.csv);;HDF5 Files (*.h5);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare export data
            export_data = {
                'tracks': self.tracking_results
            }
            
            # Add compartment data if available
            if self.compartment_results is not None:
                export_data['compartments'] = self.compartment_results
            
            # Use the IO module to export the results
            io.save_tracks(export_data, file_path)
            
            QMessageBox.information(self, "Success", "Results exported successfully")
            logger.info(f"Tracking results exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
            logger.error(f"Error exporting results: {str(e)}")
    
    def _on_test_detection(self):
        """Test particle detection on the current frame."""
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "No image data loaded")
            return
        
        try:
            # Get settings from UI
            method = self.detection_method.currentText()
            threshold = self.threshold.value()
            min_distance = self.min_distance.value()
            diameter = self.diameter.value()
            subpixel = self.subpixel.isChecked()
            
            # Additional parameters for advanced methods
            params = {}
            if method == "unet":
                params['model_path'] = self.model_path
                params['use_gpu'] = self.use_gpu.isChecked()
                params['confidence_threshold'] = self.confidence_threshold.value()
            
            # Create detector object
            det = detector.ParticleDetector(method=method)
            
            # Run detection on first frame
            first_frame = self.image_data[0] if isinstance(self.image_data, list) else self.image_data
            spots = det.detect(
                first_frame, 
                threshold=threshold,
                min_distance=min_distance,
                diameter=diameter,
                subpixel=subpixel,
                **params
            )
            
            # Display results
            self._display_detection_results(first_frame, spots)
            
            logger.info(f"Detection test complete: found {len(spots)} particles")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            logger.error(f"Error during detection test: {str(e)}")
    
    def _on_test_segmentation(self):
        """Test compartment segmentation on the current frame."""
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "No image data loaded")
            return
        
        if not self.compartment_definitions:
            QMessageBox.warning(self, "Warning", "No compartments defined")
            return
        
        try:
            # Get segmentation parameters
            method = self.segmentation_method.currentText()
            apply_preprocessing = self.apply_preprocessing.isChecked()
            fill_holes = self.fill_holes.isChecked()
            smooth_boundaries = self.smooth_boundaries.isChecked()
            
            # Create compartment analyzer
            comp_analyzer = compartment_analyzer.CompartmentAnalyzer(
                method=method.lower().replace(" ", "_"),
                apply_preprocessing=apply_preprocessing,
                fill_holes=fill_holes,
                smooth_boundaries=smooth_boundaries
            )
            
            # Get first frame
            first_frame = self.image_data[0] if isinstance(self.image_data, list) else self.image_data
            
            # If multi-channel, ensure we have enough channels
            if isinstance(first_frame, np.ndarray) and first_frame.ndim > 2:
                required_channels = max([comp.channel for comp in self.compartment_definitions]) + 1
                if first_frame.shape[0] < required_channels:
                    QMessageBox.warning(
                        self, 
                        "Warning", 
                        f"Image has {first_frame.shape[0]} channels, but compartment definitions require {required_channels}"
                    )
                    return
            
            # Segment compartments
            compartments = comp_analyzer.segment_compartments(
                first_frame,
                self.compartment_definitions
            )
            
            # Display results
            self._display_segmentation_results(first_frame, compartments)
            
            logger.info(f"Segmentation test complete: found {len(compartments)} compartments")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Segmentation failed: {str(e)}")
            logger.error(f"Error during segmentation test: {str(e)}")
    
    def _on_run_tracking(self):
        """Run the full tracking and analysis pipeline on the image data."""
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "No image data loaded")
            return
        
        try:
            # Get all settings from UI
            detection_params = {
                'method': self.detection_method.currentText(),
                'threshold': self.threshold.value(),
                'min_distance': self.min_distance.value(),
                'diameter': self.diameter.value(),
                'subpixel': self.subpixel.isChecked()
            }
            
            # Additional parameters for advanced methods
            if detection_params['method'] == "unet":
                detection_params['model_path'] = self.model_path
                detection_params['use_gpu'] = self.use_gpu.isChecked()
                detection_params['confidence_threshold'] = self.confidence_threshold.value()
            
            linker_params = {
                'method': self.linking_method.currentText(),
                'max_distance': self.max_distance.value(),
                'max_gap_frames': self.max_gap_frames.value(),
                'memory': self.memory.value()
            }
            
            tracker_params = {
                'min_track_length': self.min_track_length.value(),
                'pixel_size': self.pixel_size.value(),
                'frame_interval': self.frame_interval.value()
            }
            
            multi_channel_params = {
                'enabled': self.enable_multi_channel.isChecked(),
                'channel_count': self.channel_count.value(),
                'reference_channel': self.reference_channel.value()
            }
            
            compartment_params = {
                'enabled': self.enable_compartment.isChecked(),
                'method': self.segmentation_method.currentText(),
                'apply_preprocessing': self.apply_preprocessing.isChecked(),
                'fill_holes': self.fill_holes.isChecked(),
                'smooth_boundaries': self.smooth_boundaries.isChecked(),
                'compartments': self.compartment_definitions
            }
            
            # Update status and show progress bar
            self.status_label.setText("Running tracking... Please wait")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.run_tracking_button.setEnabled(False)
            self.test_detection_button.setEnabled(False)
            self.test_segmentation_button.setEnabled(False)
            QApplication.processEvents()  # Allow UI to update
            
            # Create detector object
            det = detector.ParticleDetector(method=detection_params['method'])
            
            # Create linker object
            link = linker.ParticleLinker(method=linker_params['method'])
            
            # Create tracker object
            track = tracker.Tracker(
                min_track_length=tracker_params['min_track_length'],
                pixel_size=tracker_params['pixel_size'],
                frame_interval=tracker_params['frame_interval']
            )
            
            # Run detection
            if isinstance(self.image_data, list):
                spots_list = []
                total_frames = len(self.image_data)
                
                for i, frame in enumerate(self.image_data):
                    # Advanced params for specific methods
                    extra_params = {}
                    if detection_params['method'] == "unet":
                        extra_params['model_path'] = detection_params['model_path']
                        extra_params['use_gpu'] = detection_params['use_gpu']
                        extra_params['confidence_threshold'] = detection_params['confidence_threshold']
                    
                    spots = det.detect(
                        frame, 
                        threshold=detection_params['threshold'],
                        min_distance=detection_params['min_distance'],
                        diameter=detection_params['diameter'],
                        subpixel=detection_params['subpixel'],
                        **extra_params
                    )
                    spots_list.append(spots)
                    
                    # Update progress
                    progress = int((i + 1) / total_frames * 50)  # First 50% for detection
                    self.progress_bar.setValue(progress)
                    QApplication.processEvents()
                
                # Link spots into tracks
                self.status_label.setText("Linking particles... Please wait")
                tracks_data = link.link_particles(
                    spots_list,
                    max_distance=linker_params['max_distance'],
                    max_gap_frames=linker_params['max_gap_frames'],
                    memory=linker_params['memory']
                )
                self.progress_bar.setValue(75)  # 75% for linking
                QApplication.processEvents()
                
            else:
                # Single frame case
                extra_params = {}
                if detection_params['method'] == "unet":
                    extra_params['model_path'] = detection_params['model_path']
                    extra_params['use_gpu'] = detection_params['use_gpu']
                    extra_params['confidence_threshold'] = detection_params['confidence_threshold']
                
                spots = det.detect(
                    self.image_data, 
                    threshold=detection_params['threshold'],
                    min_distance=detection_params['min_distance'],
                    diameter=detection_params['diameter'],
                    subpixel=detection_params['subpixel'],
                    **extra_params
                )
                tracks_data = spots  # No linking for single frame
                self.progress_bar.setValue(75)
                QApplication.processEvents()
            
            # Process tracks
            self.status_label.setText("Processing tracks... Please wait")
            self.tracking_results = track.process_tracks(tracks_data)
            
            # Handle multi-channel if enabled
            if multi_channel_params['enabled'] and multi_channel_params['channel_count'] > 1:
                mc_manager = multi_channel.MultiChannelManager(
                    channel_count=multi_channel_params['channel_count'],
                    reference_channel=multi_channel_params['reference_channel']
                )
                self.tracking_results = mc_manager.align_channels(self.tracking_results)
            
            # Process compartments if enabled
            self.compartment_results = None
            if compartment_params['enabled'] and compartment_params['compartments']:
                self.status_label.setText("Analyzing compartments... Please wait")
                
                # Create compartment analyzer
                comp_analyzer = compartment_analyzer.CompartmentAnalyzer(
                    method=compartment_params['method'].lower().replace(" ", "_"),
                    apply_preprocessing=compartment_params['apply_preprocessing'],
                    fill_holes=compartment_params['fill_holes'],
                    smooth_boundaries=compartment_params['smooth_boundaries']
                )
                
                # Segment compartments for all frames
                all_compartments = []
                
                if isinstance(self.image_data, list):
                    for i, frame in enumerate(self.image_data):
                        compartments = comp_analyzer.segment_compartments(
                            frame,
                            compartment_params['compartments']
                        )
                        all_compartments.append(compartments)
                        
                        # Update progress (last 10%)
                        progress = 90 + int((i + 1) / len(self.image_data) * 10)
                        self.progress_bar.setValue(progress)
                        QApplication.processEvents()
                else:
                    compartments = comp_analyzer.segment_compartments(
                        self.image_data,
                        compartment_params['compartments']
                    )
                    all_compartments.append(compartments)
                
                # Analyze tracks in relation to compartments
                self.compartment_results = comp_analyzer.analyze_tracks_in_compartments(
                    self.tracking_results,
                    all_compartments
                )
                
                # Enable compartment visualization
                self.show_compartments_button.setEnabled(True)
            
            # Update status and enable export and visualization
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Tracking complete: {len(self.tracking_results)} tracks")
            self.progress_bar.setVisible(False)
            self.export_button.setEnabled(True)
            self.show_tracks_button.setEnabled(True)
            self.show_histogram_button.setEnabled(True)
            self.save_figure_button.setEnabled(True)
            self.run_tracking_button.setEnabled(True)
            self.test_detection_button.setEnabled(True)
            if self.enable_compartment.isChecked():
                self.test_segmentation_button.setEnabled(True)
            
            # Show tracks
            self._on_show_tracks()
            
            # Emit signal that tracking is complete
            self.tracking_complete.emit({
                'tracks': self.tracking_results,
                'compartments': self.compartment_results
            })
            
            logger.info(f"Tracking complete: {len(self.tracking_results)} tracks generated")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed: {str(e)}")
            self.status_label.setText("Tracking failed")
            self.progress_bar.setVisible(False)
            self.run_tracking_button.setEnabled(True)
            self.test_detection_button.setEnabled(True)
            if self.enable_compartment.isChecked():
                self.test_segmentation_button.setEnabled(True)
            logger.error(f"Error during tracking: {str(e)}")
    
    def _display_frame(self, frame_index=0):
        """Display a specific frame in the visualization area."""
        if self.image_data is None:
            return
        
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get the frame data
        if isinstance(self.image_data, list):
            frame = self.image_data[frame_index]
        else:
            frame = self.image_data  # Single frame
        
        # Handle multi-channel data
        if isinstance(frame, np.ndarray) and frame.ndim > 2:
            # For multi-channel data, display the first channel or particle channel
            if self.enable_multi_channel.isChecked():
                ref_channel = self.reference_channel.value()
                if ref_channel < frame.shape[0]:
                    frame_to_display = frame[ref_channel]
                else:
                    frame_to_display = frame[0]
            else:
                frame_to_display = frame[0]
        else:
            frame_to_display = frame
        
        # Display the image
        ax.imshow(frame_to_display, cmap='gray')
        ax.set_title(f"Frame {frame_index}")
        ax.axis('off')
        
        self.canvas.draw()
    
    def _display_detection_results(self, frame, spots):
        """Display detection results on the current frame."""
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Handle multi-channel data for display
        if isinstance(frame, np.ndarray) and frame.ndim > 2:
            if self.enable_multi_channel.isChecked():
                ref_channel = self.reference_channel.value()
                if ref_channel < frame.shape[0]:
                    frame_to_display = frame[ref_channel]
                else:
                    frame_to_display = frame[0]
            else:
                frame_to_display = frame[0]
        else:
            frame_to_display = frame
        
        # Display the image
        ax.imshow(frame_to_display, cmap='gray')
        
        # Plot detected spots
        if spots is not None and len(spots) > 0:
            x = spots['x']
            y = spots['y']
            ax.plot(x, y, 'ro', markersize=3)
            ax.set_title(f"Detected {len(spots)} particles")
        else:
            ax.setTitle("No particles detected")
            ax.set_title("No particles detected")
        
        ax.axis('off')
        self.canvas.draw()
    
    def _display_segmentation_results(self, frame, compartments):
        """Display segmentation results on the current frame."""
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Handle multi-channel data for display
        if isinstance(frame, np.ndarray) and frame.ndim > 2:
            if self.enable_multi_channel.isChecked():
                ref_channel = self.reference_channel.value()
                if ref_channel < frame.shape[0]:
                    frame_to_display = frame[ref_channel]
                else:
                    frame_to_display = frame[0]
            else:
                frame_to_display = frame[0]
        else:
            frame_to_display = frame
        
        # Display the image
        ax.imshow(frame_to_display, cmap='gray')
        
        # Overlay compartments with semi-transparency
        for compartment_name, mask in compartments.items():
            # Find the color for this compartment
            color = None
            for comp in self.compartment_definitions:
                if comp.name == compartment_name:
                    color = comp.color
                    break
            
            if color is None:
                color = QColor(0, 255, 0, 128)
            
            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            rgba[mask > 0, 0] = color.red()
            rgba[mask > 0, 1] = color.green()
            rgba[mask > 0, 2] = color.blue()
            rgba[mask > 0, 3] = color.alpha()
            
            ax.imshow(rgba, alpha=0.5)
        
        ax.set_title(f"Segmented {len(compartments)} compartments")
        ax.axis('off')
        self.canvas.draw()
    
    def _on_show_tracks(self):
        """Display the tracking results."""
        if self.tracking_results is None or self.image_data is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if isinstance(self.image_data, list):
            background = self.image_data[0]
        else:
            background = self.image_data
        
        if isinstance(background, np.ndarray) and background.ndim > 2:
            if self.enable_multi_channel.isChecked():
                ref_channel = self.reference_channel.value()
                if ref_channel < background.shape[0]:
                    background_to_display = background[ref_channel]
                else:
                    background_to_display = background[0]
            else:
                background_to_display = background[0]
        else:
            background_to_display = background
        
        ax.imshow(background_to_display, cmap='gray', alpha=0.7)
        
        visualization.plot_tracks(self.tracking_results, ax=ax)
        
        ax.set_title(f"Tracks: {len(self.tracking_results)}")
        ax.axis('off')
        
        self.canvas.draw()
    
    def _on_show_histogram(self):
        """Display histogram of track properties."""
        if self.tracking_results is None:
            return
        
        track_lengths = [len(track) for track in self.tracking_results]
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.hist(track_lengths, bins=20, alpha=0.7)
        ax.set_xlabel('Track Length (frames)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Track Lengths')
        
        self.canvas.draw()
    
    def _on_show_compartments(self):
        """Display compartment analysis results."""
        if self.compartment_results is None or self.image_data is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if isinstance(self.image_data, list):
            background = self.image_data[0]
        else:
            background = self.image_data
        
        if isinstance(background, np.ndarray) and background.ndim > 2:
            if self.enable_multi_channel.isChecked():
                ref_channel = self.reference_channel.value()
                if ref_channel < background.shape[0]:
                    background_to_display = background[ref_channel]
                else:
                    background_to_display = background[0]
            else:
                background_to_display = background[0]
        else:
            background_to_display = background
        
        ax.imshow(background_to_display, cmap='gray', alpha=0.7)
        
        if isinstance(self.compartment_results, list):
            compartments = self.compartment_results[0]['compartments']
        else:
            compartments = self.compartment_results['compartments']
        
        for compartment_name, mask in compartments.items():
            color = None
            for comp in self.compartment_definitions:
                if comp.name == compartment_name:
                    color = comp.color
                    break
            
            if color is None:
                color = QColor(0, 255, 0, 128)
            
            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            rgba[mask > 0, 0] = color.red()
            rgba[mask > 0, 1] = color.green()
            rgba[mask > 0, 2] = color.blue()
            rgba[mask > 0, 3] = color.alpha()
            
            ax.imshow(rgba, alpha=0.5)
        
        visualization.plot_tracks(self.tracking_results, ax=ax)
        
        track_counts = {}
        for comp_name in compartments.keys():
            if isinstance(self.compartment_results, list):
                track_counts[comp_name] = sum(
                    1 for frame_result in self.compartment_results 
                    for track_id in frame_result.get('track_compartments', {}).keys()
                    if frame_result.get('track_compartments', {}).get(track_id, '') == comp_name
                )
            else:
                track_counts[comp_name] = sum(
                    1 for track_id, comp in self.compartment_results.get('track_compartments', {}).items()
                    if comp == comp_name
                )
        
        legend_text = [f"{comp}: {count} tracks" for comp, count in track_counts.items()]
        ax.legend(legend_text, loc='upper right')
        
        ax.set_title("Tracks in Compartments")
        ax.axis('off')
        
        self.canvas.draw()
    
    def _on_save_figure(self):
        """Save the current figure to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Figure", 
            "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", "Figure saved successfully")
            logger.info(f"Figure saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save figure: {str(e)}")
            logger.error(f"Error saving figure: {str(e)}")
    
    def load_project_data(self, project_data):
        """
        Load data from a project.
        
        Parameters
        ----------
        project_data : dict
            Project data containing tracking configuration and results
        """
        if 'tracking_config' in project_data:
            config = project_data['tracking_config']
            
            if 'detector' in config:
                det = config['detector']
                if 'method' in det:
                    index = self.detection_method.findText(det['method'])
                    if index >= 0:
                        self.detection_method.setCurrentIndex(index)
                # Update other detector parameters...
            
            if 'linker' in config:
                link = config['linker']
                if 'method' in link:
                    index = self.linking_method.findText(link['method'])
                    if index >= 0:
                        self.linking_method.setCurrentIndex(index)
                if 'max_distance' in link:
                    self.max_distance.setValue(link['max_distance'])
                if 'max_gap_frames' in link:
                    self.max_gap_frames.setValue(link['max_gap_frames'])
                if 'memory' in link:
                    self.memory.setValue(link['memory'])
            
            if 'tracker' in config:
                track = config['tracker']
                if 'min_track_length' in track:
                    self.min_track_length.setValue(track['min_track_length'])
                if 'pixel_size' in track:
                    self.pixel_size.setValue(track['pixel_size'])
                if 'frame_interval' in track:
                    self.frame_interval.setValue(track['frame_interval'])
            
            if 'multi_channel' in config:
                mc = config['multi_channel']
                if 'enabled' in mc:
                    self.enable_multi_channel.setChecked(mc['enabled'])
                if 'channel_count' in mc:
                    self.channel_count.setValue(mc['channel_count'])
                if 'reference_channel' in mc:
                    self.reference_channel.setValue(mc['reference_channel'])
            
            if 'compartment_analysis' in config:
                comp = config['compartment_analysis']
                if 'enabled' in comp:
                    self.enable_compartment.setChecked(comp['enabled'])
                if 'segmentation_method' in comp:
                    index = self.segmentation_method.findText(comp['segmentation_method'])
                    if index >= 0:
                        self.segmentation_method.setCurrentIndex(index)
                if 'apply_preprocessing' in comp:
                    self.apply_preprocessing.setChecked(comp['apply_preprocessing'])
                if 'fill_holes' in comp:
                    self.fill_holes.setChecked(comp['fill_holes'])
                if 'smooth_boundaries' in comp:
                    self.smooth_boundaries.setChecked(comp['smooth_boundaries'])
                
                # Load compartment definitions
                if 'compartments' in comp:
                    self.compartment_definitions = []
                    self.compartment_list.clear()
                    
                    for comp_dict in comp['compartments']:
                        try:
                            compartment = compartment_analyzer.CompartmentDefinition.from_dict(comp_dict)
                            self.compartment_definitions.append(compartment)
                            self.compartment_list.addItem(compartment.name)
                        except Exception as e:
                            logger.warning(f"Failed to load compartment: {str(e)}")
            
        if 'image_data' in project_data:
            self.image_data = project_data['image_data']
            self.test_detection_button.setEnabled(True)
            self.run_tracking_button.setEnabled(True)
            if self.enable_compartment.isChecked():
                self.test_segmentation_button.setEnabled(True)
            self._display_frame(0)
        
        if 'tracking_results' in project_data:
            self.tracking_results = project_data['tracking_results']
            self.export_button.setEnabled(True)
            self.show_tracks_button.setEnabled(True)
            self.show_histogram_button.setEnabled(True)
            self.save_figure_button.setEnabled(True)
            self._on_show_tracks()
            
        if 'compartment_results' in project_data:
            self.compartment_results = project_data['compartment_results']
            if self.enable_compartment.isChecked():
                self.show_compartments_button.setEnabled(True)
        
        if 'compartment_definitions' in project_data:
            self.compartment_definitions = project_data['compartment_definitions']
            self.compartment_list.clear()
            for comp in self.compartment_definitions:
                self.compartment_list.addItem(comp.name)
                
        logger.info("Project data loaded into tracking panel")
    
    def get_project_data(self):
        """
        Get data for project saving.
        
        Returns
        -------
        dict
            Dictionary containing tracking configuration and results
        """
        project_data = {}
        
        project_data['tracking_config'] = {
            'detector': {
                'method': self.detection_method.currentText(),
                'threshold': self.threshold.value(),
                'min_distance': self.min_distance.value(),
                'diameter': self.diameter.value(),
                'subpixel': self.subpixel.isChecked(),
                'model_path': self.model_path,
                'use_gpu': self.use_gpu.isChecked(),
                'confidence_threshold': self.confidence_threshold.value()
            },
            'linker': {
                'method': self.linking_method.currentText(),
                'max_distance': self.max_distance.value(),
                'max_gap_frames': self.max_gap_frames.value(),
                'memory': self.memory.value()
            },
            'tracker': {
                'min_track_length': self.min_track_length.value(),
                'pixel_size': self.pixel_size.value(),
                'frame_interval': self.frame_interval.value()
            },
            'multi_channel': {
                'enabled': self.enable_multi_channel.isChecked(),
                'channel_count': self.channel_count.value(),
                'reference_channel': self.reference_channel.value()
            },
            'compartment_analysis': {
                'enabled': self.enable_compartment.isChecked(),
                'segmentation_method': self.segmentation_method.currentText(),
                'apply_preprocessing': self.apply_preprocessing.isChecked(),
                'fill_holes': self.fill_holes.isChecked(),
                'smooth_boundaries': self.smooth_boundaries.isChecked()
            }
        }
        
        if self.tracking_results is not None:
            project_data['tracking_results'] = self.tracking_results
        
        if self.compartment_results is not None:
            project_data['compartment_results'] = self.compartment_results
        
        if self.compartment_definitions:
            project_data['compartment_definitions'] = self.compartment_definitions
        
        return project_data
