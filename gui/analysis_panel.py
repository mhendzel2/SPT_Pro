"""
Analysis panel module for SPT Analysis GUI.

This module provides the analysis panel for the SPT Analysis GUI,
with controls for various analysis types.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QCheckBox, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import logging

logger = logging.getLogger(__name__)


class AnalysisPanel(QWidget):
    """
    Panel for configuring and running different analysis types.
    
    This panel provides controls for selecting and configuring different
    analysis methods, and displaying the results.
    """
    
    # Signal emitted when analysis is complete
    analysis_complete = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up layout
        self.layout = QVBoxLayout(self)
        
        # Add analysis type selector
        self.add_analysis_selector()
        
        # Add analysis options
        self.add_analysis_options()
        
        # Add parameter controls
        self.add_parameter_controls()
        
        # Add run button
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.run_btn)
        
        # Add progress bar
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        
        # Initialize analysis objects
        self.tracks_df = None
        self.analysis_results = {}
        # Assume compartment masks may be set externally (required for some analysis types)
        self.compartment_masks = {}
    
    def add_analysis_selector(self):
        """Add the analysis type selector."""
        group = QGroupBox("Analysis Type")
        group_layout = QVBoxLayout()
        
        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            'Diffusion Analysis',
            'Dwell Time Analysis',
            'Gel Structure Analysis',
            'Diffusion Population Analysis',
            'Crowding Analysis',
            'Active Transport Analysis',
            'Boundary Crossing Analysis'
        ])
        self.analysis_type.currentIndexChanged.connect(self.update_analysis_options)
        
        group_layout.addWidget(self.analysis_type)
        group.setLayout(group_layout)
        self.layout.addWidget(group)
    
    def add_analysis_options(self):
        """Add the analysis-specific options."""
        self.options_group = QGroupBox("Analysis Options")
        self.options_layout = QVBoxLayout()
        
        # Create stack of option widgets
        self.diffusion_options = self.create_diffusion_options()
        self.dwell_time_options = self.create_dwell_time_options()
        self.gel_structure_options = self.create_gel_structure_options()
        self.diffusion_population_options = self.create_diffusion_population_options()
        self.crowding_options = self.create_crowding_options()
        self.active_transport_options = self.create_active_transport_options()
        self.boundary_crossing_options = self.create_boundary_crossing_options()
        
        # Add initial options widget
        self.options_layout.addWidget(self.diffusion_options)
        
        # Hide other options widgets
        self.dwell_time_options.hide()
        self.gel_structure_options.hide()
        self.diffusion_population_options.hide()
        self.crowding_options.hide()
        self.active_transport_options.hide()
        self.boundary_crossing_options.hide()
        
        self.options_group.setLayout(self.options_layout)
