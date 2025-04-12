"""
Tracking panel module for SPT Analysis GUI.

This module provides the tracking panel for the SPT Analysis GUI, with controls for
particle detection, linking, and tracking operations.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, 
    QCheckBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget, 
    QTextEdit, QFileDialog, QMessageBox, QProgressBar, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import os
import logging

from ..tracking import detector, linker, tracker, multi_channel
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
    - Visualizing tracking results
    """
    
    # Signal emitted when tracking is complete
    tracking_complete = pyqtSignal(object)
    
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
        self.detection_method.addItems(["gaussian", "laplacian", "doh", "log"])
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
        
        self.save_figure_button = QPushButton("Save Figure")
        self.save_figure_button.clicked.connect(self._on_save_figure)
        self.save_figure_button.setEnabled(False)
        
        viz_buttons_layout.addWidget(self.show_tracks_button)
        viz_buttons_layout.addWidget(self.show_histogram_button)
        viz_buttons_layout.addWidget(self.save_figure_button)
        viz_buttons_layout.addStretch()
        
        viz_layout.addLayout(viz_buttons_layout)
        viz_group.setLayout(viz_layout)
        main_layout.addWidget(viz_group)
    
    def _toggle_multi_channel(self, enabled):
        """Enable or disable multi-channel settings based on checkbox state."""
        self.channel_count.setEnabled(enabled)
        self.reference_channel.setEnabled(enabled)
    
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
                    'subpixel': self.subpixel.isChecked()
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
            
            QMessageBox.information(self, "Success", "Configuration loaded successfully")
            logger.info(f"Configuration loaded from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _on_export_results(self):
        """Export tracking results to a file."""
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
            # Use the IO module to export the results
            io.save_tracks(self.tracking_results, file_path)
            
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
            
            # Create detector object
            det = detector.ParticleDetector(method=method)
            
            # Run detection on first frame
            first_frame = self.image_data[0] if isinstance(self.image_data, list) else self.image_data
            spots = det.detect(
                first_frame, 
                threshold=threshold,
                min_distance=min_distance,
                diameter=diameter,
                subpixel=subpixel
            )
            
            # Display results
            self._display_detection_results(first_frame, spots)
            
            logger.info(f"Detection test complete: found {len(spots)} particles")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            logger.error(f"Error during detection test: {str(e)}")
    
    def _on_run_tracking(self):
        """Run the full tracking pipeline on the image data."""
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
            
            # Update status and show progress bar
            self.status_label.setText("Running tracking... Please wait")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.run_tracking_button.setEnabled(False)
            self.test_detection_button.setEnabled(False)
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
                    spots = det.detect(
                        frame, 
                        threshold=detection_params['threshold'],
                        min_distance=detection_params['min_distance'],
                        diameter=detection_params['diameter'],
                        subpixel=detection_params['subpixel']
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
                spots = det.detect(
                    self.image_data, 
                    threshold=detection_params['threshold'],
                    min_distance=detection_params['min_distance'],
                    diameter=detection_params['diameter'],
                    subpixel=detection_params['subpixel']
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
            
            # Show tracks
            self._on_show_tracks()
            
            # Emit signal that tracking is complete
            self.tracking_complete.emit(self.tracking_results)
            
            logger.info(f"Tracking complete: {len(self.tracking_results)} tracks generated")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed: {str(e)}")
            self.status_label.setText("Tracking failed")
            self.progress_bar.setVisible(False)
            self.run_tracking_button.setEnabled(True)
            self.test_detection_button.setEnabled(True)
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
        
        # Display the image
        ax.imshow(frame, cmap='gray')
        ax.set_title(f"Frame {frame_index}")
        ax.axis('off')
        
        self.canvas.draw()
    
    def _display_detection_results(self, frame, spots):
        """Display detection results on the current frame."""
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Display the image
        ax.imshow(frame, cmap='gray')
        
        # Plot detected spots
        if spots is not None and len(spots) > 0:
            x = spots['x']
            y = spots['y']
            ax.plot(x, y, 'ro', markersize=3)
            ax.set_title(f"Detected {len(spots)} particles")
        else:
            ax.set_title("No particles detected")
        
        ax.axis('off')
        self.canvas.draw()
    
    def _on_show_tracks(self):
        """Display the tracking results."""
        if self.tracking_results is None or self.image_data is None:
            return
        
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get background image (first frame)
        if isinstance(self.image_data, list):
            background = self.image_data[0]
        else:
            background = self.image_data
        
        # Display the image
        ax.imshow(background, cmap='gray', alpha=0.7)
        
        # Visualize tracks using the visualization module
        visualization.plot_tracks(self.tracking_results, ax=ax)
        
        ax.set_title(f"Tracks: {len(self.tracking_results)}")
        ax.axis('off')
        
        self.canvas.draw()
    
    def _on_show_histogram(self):
        """Display histogram of track properties."""
        if self.tracking_results is None:
            return
        
        # Calculate track lengths
        track_lengths = [len(track) for track in self.tracking_results]
        
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot histogram
        ax.hist(track_lengths, bins=20, alpha=0.7)
        ax.set_xlabel('Track Length (frames)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Track Lengths')
        
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
            # Update UI with project configuration
            config = project_data['tracking_config']
            
            if 'detector' in config:
                det = config['detector']
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
        
        if 'image_data' in project_data:
            self.image_data = project_data['image_data']
            self.test_detection_button.setEnabled(True)
            self.run_tracking_button.setEnabled(True)
            self._display_frame(0)
        
        if 'tracking_results' in project_data:
            self.tracking_results = project_data['tracking_results']
            self.export_button.setEnabled(True)
            self.show_tracks_button.setEnabled(True)
            self.show_histogram_button.setEnabled(True)
            self.save_figure_button.setEnabled(True)
            self._on_show_tracks()
            
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
        
        # Save configuration
        project_data['tracking_config'] = {
            'detector': {
                'method': self.detection_method.currentText(),
                'threshold': self.threshold.value(),
                'min_distance': self.min_distance.value(),
                'diameter': self.diameter.value(),
                'subpixel': self.subpixel.isChecked()
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
            }
        }
        
        # Save results (if available)
        if self.tracking_results is not None:
            project_data['tracking_results'] = self.tracking_results
        
        # Typically, image_data is not saved directly in project files
        
        return project_data
