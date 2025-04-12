
"""
Main window module for the SPT Analysis GUI.

This module provides a graphical user interface for the SPT Analysis package using PyQt5.
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import pandas as pd
import numpy as np
import os
import sys
import logging

from ..tracking import tracker
from ..analysis import diffusion, motion, clustering
from ..utils import io, processing
from ..project import management

logger = logging.getLogger(__name__)


class EnhancedTrackingAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced SPT Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create side panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # Add controls
        self.add_file_controls(control_layout)
        self.add_preprocessing_controls(control_layout)
        self.add_analysis_controls(control_layout)
        self.add_plot_controls(control_layout)

        # Add progress bar
        self.progress = QProgressBar()
        control_layout.addWidget(self.progress)

        # Create right panel for results
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)

        # Add matplotlib canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        results_layout.addWidget(self.canvas)

        # Add results text box
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        # Add panels to main layout
        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(results_panel, stretch=2)
        
        # Initialize data store
        self.tracks_df = None
        self.project = None
        self.current_cell = None

    def add_file_controls(self, layout):
        group = QGroupBox("File Controls")
        group_layout = QVBoxLayout()

        self.load_btn = QPushButton("Load Files")
        self.load_btn.clicked.connect(self.load_files)
        group_layout.addWidget(self.load_btn)
        
        self.project_btn = QPushButton("Create/Load Project")
        self.project_btn.clicked.connect(self.manage_project)
        group_layout.addWidget(self.project_btn)

        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        group_layout.addWidget(self.file_list)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def add_preprocessing_controls(self, layout):
        group = QGroupBox("Preprocessing")
        group_layout = QVBoxLayout()

        self.min_track_length = QSpinBox()
        self.min_track_length.setRange(5, 100)
        self.min_track_length.setValue(10)
        group_layout.addWidget(QLabel("Min Track Length:"))
        group_layout.addWidget(self.min_track_length)

        self.drift_correction = QCheckBox("Apply Drift Correction")
        self.drift_correction.setChecked(True)
        group_layout.addWidget(self.drift_correction)
        
        self.denoise = QCheckBox("Apply Denoising")
        self.denoise.setChecked(True)
        group_layout.addWidget(self.denoise)
        
        self.preprocess_btn = QPushButton("Preprocess Data")
        self.preprocess_btn.clicked.connect(self.preprocess_data)
        group_layout.addWidget(self.preprocess_btn)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def add_analysis_controls(self, layout):
        group = QGroupBox("Analysis Options")
        group_layout = QVBoxLayout()

        self.analysis_type = QComboBox()
        self.analysis_type.addItems(['MSD Analysis', 'Jump Size Distribution', 
                                   'Diffusion Modes', 'Rheological Properties'])
        group_layout.addWidget(self.analysis_type)
        
        # Add parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.01, 1.0)
        self.pixel_size.setValue(0.1)
        self.pixel_size.setSingleStep(0.01)
        params_layout.addRow("Pixel Size (μm):", self.pixel_size)
        
        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setRange(0.001, 10.0)
        self.frame_interval.setValue(0.1)
        self.frame_interval.setSingleStep(0.01)
        params_layout.addRow("Frame Interval (s):", self.frame_interval)
        
        params_group.setLayout(params_layout)
        group_layout.addWidget(params_group)

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        group_layout.addWidget(self.run_btn)
        
        # Add population analysis button
        self.population_btn = QPushButton("Population Analysis")
        self.population_btn.clicked.connect(self.run_population_analysis)
        group_layout.addWidget(self.population_btn)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def add_plot_controls(self, layout):
        group = QGroupBox("Plot Options")
        group_layout = QVBoxLayout()

        self.plot_style = QComboBox()
        self.plot_style.addItems(['default', 'seaborn', 'ggplot'])
        self.plot_style.currentTextChanged.connect(self.update_plot_style)
        group_layout.addWidget(QLabel("Plot Style:"))
        group_layout.addWidget(self.plot_style)

        self.error_bars = QCheckBox("Show Error Bars")
        self.error_bars.setChecked(True)
        group_layout.addWidget(self.error_bars)
        
        # Add plot type selection
        self.plot_type = QComboBox()
        self.plot_type.addItems(['Trajectories', 'MSD Curve', 'Diffusion Coefficient', 
                               'Jump Distribution', 'Cluster Map'])
        group_layout.addWidget(QLabel("Plot Type:"))
        group_layout.addWidget(self.plot_type)
        
        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.generate_plot)
        group_layout.addWidget(self.plot_btn)
        
        self.save_plot_btn = QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self.save_plot)
        group_layout.addWidget(self.save_plot_btn)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def load_files(self):
        """Load track files and display in file list."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", 
                                              "Track Files (*.csv *.xlsx *.h5 *.json)")
        if not files:
            return
            
        self.file_list.clear()
        self.files = files
        
        # Add files to list
        for file in files:
            self.file_list.addItem(os.path.basename(file))
        
        # Select first file
        self.file_list.setCurrentRow(0)
        
        # Update status
        self.results_text.append(f"Loaded {len(files)} track files.")
    
    def on_file_selected(self):
        """Handle file selection from the list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
            
        selected_file = selected_items[0].text()
        file_idx = self.file_list.currentRow()
        file_path = self.files[file_idx]
        
        try:
            # Load tracks
            self.tracks_df = io.load_tracks(file_path)
            
            # Update display
            self.results_text.append(f"Selected file: {selected_file}")
            self.results_text.append(f"Loaded {len(self.tracks_df)} track points in "
                                   f"{self.tracks_df['track_id'].nunique()} tracks.")
            
            # Plot trajectories
            self.plot_trajectories()
            
        except Exception as e:
            self.results_text.append(f"Error loading file: {str(e)}")
    
    def manage_project(self):
        """Create or load a project."""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Project Management")
        dialog.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add options
        create_btn = QPushButton("Create New Project")
        load_btn = QPushButton("Load Existing Project")
        
        layout.addWidget(create_btn)
        layout.addWidget(load_btn)
        
        # Connect events
        create_btn.clicked.connect(lambda: self.create_project(dialog))
        load_btn.clicked.connect(lambda: self.load_project(dialog))
        
        dialog.exec_()
    
    def create_project(self, parent_dialog=None):
        """Create a new project."""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Project")
        dialog.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add fields
        form_layout = QFormLayout()
        
        name_input = QLineEdit()
        description_input = QTextEdit()
        description_input.setMaximumHeight(100)
        
        form_layout.addRow("Project Name:", name_input)
        form_layout.addRow("Description:", description_input)
        
        layout.addLayout(form_layout)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        layout.addWidget(button_box)
        
        # Close parent dialog if provided
        if parent_dialog:
            parent_dialog.close()
        
        # Execute dialog
        if dialog.exec_() == QDialog.Accepted:
            project_name = name_input.text()
            description = description_input.toPlainText()
            
            if not project_name:
                QMessageBox.warning(self, "Warning", "Project name is required.")
                return
            
            # Create project
            self.project = management.SPTProject(project_name, description)
            
            # Update status
            self.results_text.append(f"Created new project: {project_name}")
            self.results_text.append(f"Project ID: {self.project.project_id}")
    
    def load_project(self, parent_dialog=None):
        """Load an existing project."""
        # Get file path
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", 
                                                "SPT Project Files (*.spt)")
        
        if not file_path:
            return
        
        try:
            # Load project
            self.project = management.SPTProject.load_project(file_path)
            
            # Close parent dialog if provided
            if parent_dialog:
                parent_dialog.close()
            
            # Update status
            self.results_text.append(f"Loaded project: {self.project.name}")
            self.results_text.append(f"Project contains {len(self.project.treatments)} treatment groups.")
            
            # Update file list with cells
            self.file_list.clear()
            
            for treatment_name, treatment in self.project.treatments.items():
                for cell_id, cell in treatment.cells.items():
                    self.file_list.addItem(f"{treatment_name} - {cell_id}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading project: {str(e)}")
    
    def preprocess_data(self):
        """Preprocess the selected track data."""
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load a file first.")
            return
        
        try:
            # Apply preprocessing steps
            preprocessed_df = self.tracks_df.copy()
            
            # Filter by track length
            min_length = self.min_track_length.value()
            track_lengths = preprocessed_df.groupby('track_id').size()
            valid_tracks = track_lengths[track_lengths >= min_length].index
            preprocessed_df = preprocessed_df[preprocessed_df['track_id'].isin(valid_tracks)]
            
            # Apply drift correction if selected
            if self.drift_correction.isChecked():
                # Calculate drift per frame
                drift = preprocessed_df.groupby('frame')[['x', 'y']].mean()
                drift = drift - drift.iloc[0]  # Reference to first frame
                
                # Correct positions
                for frame in preprocessed_df['frame'].unique():
                    mask = preprocessed_df['frame'] == frame
                    preprocessed_df.loc[mask, 'x'] -= drift.loc[frame, 'x']
                    preprocessed_df.loc[mask, 'y'] -= drift.loc[frame, 'y']
            
            # Update tracks_df
            self.tracks_df = preprocessed_df
            
            # Update display
            self.results_text.append(f"Preprocessing complete. Retained {len(self.tracks_df)} track points in "
                                   f"{self.tracks_df['track_id'].nunique()} tracks.")
            
            # Update plot
            self.plot_trajectories()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during preprocessing: {str(e)}")
    
    def run_analysis(self):
        """Run the selected analysis on the current data."""
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load a file first.")
            return
        
        analysis_type = self.analysis_type.currentText()
        pixel_size = self.pixel_size.value()
        frame_interval = self.frame_interval.value()
        
        try:
            self.progress.setValue(10)
            
            # Perform analysis based on selected type
            if analysis_type == 'MSD Analysis':
                # Create diffusion analyzer
                analyzer = diffusion.DiffusionAnalyzer(
                    pixel_size=pixel_size,
                    frame_interval=frame_interval
                )
                
                self.progress.setValue(30)
                
                # Compute MSD
                msd_df, ensemble_df = analyzer.compute_msd(
                    self.tracks_df,
                    max_lag=20,
                    min_track_length=self.min_track_length.value()
                )
                
                self.progress.setValue(60)
                
                # Fit diffusion models
                fit_results = analyzer.fit_diffusion_models(ensemble_df)
                
                self.progress.setValue(80)
                
                # Display results
                self.results_text.append("\n=== MSD Analysis Results ===")
                self.results_text.append(f"Diffusion coefficient: {fit_results['simple_diffusion']['D']:.4f} μm²/s")
                self.results_text.append(f"R-squared: {fit_results['simple_diffusion']['r_squared']:.4f}")
                self.results_text.append(f"Best model: {fit_results['best_model']}")
                
                if fit_results['best_model'] == 'anomalous_diffusion':
                    alpha = fit_results['anomalous_diffusion']['alpha']
                    self.results_text.append(f"Anomalous exponent (α): {alpha:.3f}")
                    if alpha < 0.9:
                        self.results_text.append("  --> Subdiffusion (α < 0.9)")
                    elif alpha > 1.1:
                        self.results_text.append("  --> Superdiffusion (α > 1.1)")
                    else:
                        self.results_text.append("  --> Brownian diffusion (0.9 ≤ α ≤ 1.1)")
                
                # Plot MSD curve
                self.plot_msd(ensemble_df, fit_results)
                
                # Store in project if available
                if self.project is not None and self.current_cell is not None:
                    # Create analysis results for each track
                    diffusion_df = analyzer.compute_diffusion_coefficient(
                        self.tracks_df, 
                        method='individual', 
                        min_track_length=self.min_track_length.value()
                    )
                    
                    # Store in cell
                    cell = self.current_cell
                    cell.tracks_df = self.tracks_df
                    
                    # Store analysis results
                    cell.analysis_results = []
                    for _, row in diffusion_df.iterrows():
                        cell.analysis_results.append({
                            'track_id': row['track_id'],
                            'diffusion_coefficient': row['D'],
                            'r_squared': row['r_squared']
                        })
                    
                    self.results_text.append(f"Analysis results stored in project cell: {cell.cell_id}")
            
            elif analysis_type == 'Jump Size Distribution':
                # Calculate jump sizes
                jumps = []
                
                for track_id, track in self.tracks_df.groupby('track_id'):
                    track = track.sort_values('frame')
                    
                    # Skip if track is too short
                    if len(track) < 2:
                        continue
                    
                    # Calculate jumps
                    dx
Copy
                    # Calculate jumps
                    dx = np.diff(track['x']) * pixel_size
                    dy = np.diff(track['y']) * pixel_size
                    jump_sizes = np.sqrt(dx**2 + dy**2)
                    
                    jumps.extend(jump_sizes)
                
                self.progress.setValue(50)
                
                # Fit jump size distribution
                from scipy import stats
                
                # Fit Rayleigh distribution
                param = stats.rayleigh.fit(jumps)
                x = np.linspace(0, max(jumps), 100)
                pdf_rayleigh = stats.rayleigh.pdf(x, *param)
                
                # Calculate diffusion coefficient from Rayleigh fit
                D_rayleigh = param[1]**2 / (4 * frame_interval)
                
                self.progress.setValue(80)
                
                # Display results
                self.results_text.append("\n=== Jump Size Analysis Results ===")
                self.results_text.append(f"Mean jump size: {np.mean(jumps):.4f} μm")
                self.results_text.append(f"Median jump size: {np.median(jumps):.4f} μm")
                self.results_text.append(f"Diffusion coefficient (from Rayleigh fit): {D_rayleigh:.4f} μm²/s")
                
                # Plot jump size distribution
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                # Plot histogram
                hist, bins, _ = ax.hist(jumps, bins=30, density=True, alpha=0.7, label='Data')
                
                # Plot Rayleigh fit
                ax.plot(x, pdf_rayleigh, 'r-', linewidth=2, label=f'Rayleigh fit (D={D_rayleigh:.4f} μm²/s)')
                
                ax.set_xlabel('Jump Size (μm)')
                ax.set_ylabel('Probability Density')
                ax.set_title('Jump Size Distribution')
                ax.grid(alpha=0.3)
                ax.legend()
                
                self.canvas.draw()
            
            elif analysis_type == 'Diffusion Modes':
                # Create diffusion analyzer
                analyzer = diffusion.DiffusionAnalyzer(
                    pixel_size=pixel_size,
                    frame_interval=frame_interval
                )
                
                self.progress.setValue(30)
                
                # Classify tracks
                classification_df, cluster_df = analyzer.classify_tracks(
                    self.tracks_df, 
                    min_track_length=self.min_track_length.value()
                )
                
                self.progress.setValue(60)
                
                if classification_df is None or len(classification_df) == 0:
                    self.results_text.append("Not enough tracks for classification.")
                    self.progress.setValue(100)
                    return
                
                # Count modes
                mode_counts = classification_df['motion_type'].value_counts()
                
                # Display results
                self.results_text.append("\n=== Diffusion Mode Analysis Results ===")
                for mode, count in mode_counts.items():
                    percentage = 100 * count / len(classification_df)
                    self.results_text.append(f"{mode}: {count} tracks ({percentage:.1f}%)")
                
                # Calculate statistics per mode
                for mode in classification_df['motion_type'].unique():
                    mode_df = classification_df[classification_df['motion_type'] == mode]
                    mean_D = mode_df['D'].mean()
                    self.results_text.append(f"  {mode} mean D: {mean_D:.4f} μm²/s")
                
                # Plot classification
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                # Create scatter plot of diffusion coefficient vs. alpha
                colors = {'Brownian': 'blue', 'Directed': 'red', 'Confined': 'green', 'Mixed': 'purple'}
                
                for mode in classification_df['motion_type'].unique():
                    mode_df = classification_df[classification_df['motion_type'] == mode]
                    ax.scatter(mode_df['D'], mode_df['alpha'], 
                              color=colors.get(mode, 'gray'), 
                              label=mode, alpha=0.7)
                
                ax.set_xlabel('Diffusion Coefficient (μm²/s)')
                ax.set_ylabel('Anomalous Exponent (α)')
                ax.set_title('Diffusion Mode Classification')
                ax.grid(alpha=0.3)
                ax.legend()
                
                self.canvas.draw()
                
                # Store in project if available
                if self.project is not None and self.current_cell is not None:
                    # Store in cell
                    cell = self.current_cell
                    cell.tracks_df = self.tracks_df
                    
                    # Store analysis results for each track
                    cell.analysis_results = []
                    for _, row in classification_df.iterrows():
                        cell.analysis_results.append({
                            'track_id': row['track_id'],
                            'diffusion_coefficient': row['D'],
                            'alpha': row['alpha'],
                            'motion_type': row['motion_type']
                        })
                    
                    self.results_text.append(f"Analysis results stored in project cell: {cell.cell_id}")
            
            elif analysis_type == 'Rheological Properties':
                # Create diffusion analyzer
                analyzer = diffusion.DiffusionAnalyzer(
                    pixel_size=pixel_size,
                    frame_interval=frame_interval
                )
                
                self.progress.setValue(30)
                
                # Compute MSD
                msd_df, ensemble_df = analyzer.compute_msd(
                    self.tracks_df,
                    max_lag=20,
                    min_track_length=self.min_track_length.value()
                )
                
                self.progress.setValue(50)
                
                # Extract time lags and MSD values
                time_lags = ensemble_df['time_lag'].values
                msd_values = ensemble_df['msd'].values
                
                # Calculate rheological properties
                from scipy.optimize import curve_fit
                
                # Fit power law to get alpha
                def power_law(x, a, alpha):
                    return a * x**alpha
                
                try:
                    popt, _ = curve_fit(power_law, time_lags, msd_values)
                    a, alpha = popt
                    
                    # Calculate complex modulus
                    # G* = kB*T / (3*pi*a*r*Gamma(1+alpha))
                    from scipy.special import gamma
                    
                    kB = 1.38e-23  # Boltzmann constant [J/K]
                    T = 310  # Temperature [K] (body temperature)
                    r = 0.1  # Particle radius [μm]
                    r_m = r * 1e-6  # Convert to meters
                    
                    # Convert a to SI units
                    a_si = a * 1e-12  # Convert from μm^2 to m^2
                    
                    # Calculate complex modulus
                    G_complex = kB * T / (3 * np.pi * a_si * r_m * gamma(1 + alpha))
                    
                    # Convert to more suitable units (Pa)
                    G_complex_Pa = G_complex
                    
                    self.progress.setValue(80)
                    
                    # Display results
                    self.results_text.append("\n=== Rheological Analysis Results ===")
                    self.results_text.append(f"Anomalous exponent (α): {alpha:.4f}")
                    
                    if 0 < alpha < 1:
                        # Viscoelastic material
                        G_storage = G_complex_Pa * np.cos(np.pi * alpha / 2)
                        G_loss = G_complex_Pa * np.sin(np.pi * alpha / 2)
                        
                        self.results_text.append(f"Complex modulus |G*|: {G_complex_Pa:.2e} Pa")
                        self.results_text.append(f"Storage modulus G': {G_storage:.2e} Pa")
                        self.results_text.append(f"Loss modulus G\": {G_loss:.2e} Pa")
                        
                        if G_storage > G_loss:
                            self.results_text.append("Material behavior: More elastic than viscous")
                        else:
                            self.results_text.append("Material behavior: More viscous than elastic")
                    
                    # Plot rheological properties
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    
                    # Plot MSD
                    ax.loglog(time_lags, msd_values, 'o', label='MSD')
                    
                    # Plot fit
                    x_fit = np.logspace(np.log10(time_lags[0]), np.log10(time_lags[-1]), 100)
                    y_fit = power_law(x_fit, a, alpha)
                    ax.loglog(x_fit, y_fit, '-', label=f'Power law fit (α={alpha:.4f})')
                    
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('MSD (μm²)')
                    ax.set_title('Rheological Properties - MSD Analysis')
                    ax.grid(True, which="both", ls="-", alpha=0.3)
                    ax.legend()
                    
                    self.canvas.draw()
                    
                except Exception as e:
                    self.results_text.append(f"Error in rheological analysis: {str(e)}")
            
            self.progress.setValue(100)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")
            self.progress.setValue(0)
    
    def run_population_analysis(self):
        """Run population analysis across all cells in the project."""
        if self.project is None:
            QMessageBox.warning(self, "Warning", "No project loaded. Please create or load a project first.")
            return
        
        if not self.project.treatments:
            QMessageBox.warning(self, "Warning", "Project has no treatments or cells. Please add cells to the project first.")
            return
        
        try:
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Population Analysis")
            dialog.setMinimumWidth(500)
            
            # Create layout
            layout = QVBoxLayout(dialog)
            
            # Add options
            form_layout = QFormLayout()
            
            # Parameter selection
            parameter_select = QComboBox()
            parameter_select.addItems(['diffusion_coefficient', 'alpha', 'motion_type'])
            form_layout.addRow("Parameter:", parameter_select)
            
            # Treatment selection
            treatment_select = QComboBox()
            treatment_select.addItem("All Treatments")
            treatment_select.addItems(self.project.list_treatment_groups())
            form_layout.addRow("Treatment:", treatment_select)
            
            # Subpopulation detection
            subpop_check = QCheckBox("Detect Subpopulations")
            form_layout.addRow("", subpop_check)
            
            layout.addLayout(form_layout)
            
            # Add buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            layout.addWidget(button_box)
            
            # Execute dialog
            if dialog.exec_() == QDialog.Accepted:
                parameter = parameter_select.currentText()
                treatment = None if treatment_select.currentIndex() == 0 else treatment_select.currentText()
                detect_subpop = subpop_check.isChecked()
                
                # Start analysis
                self.progress.setValue(10)
                
                # Pool results
                pooled_results = self.project.pool_analysis_results(parameter, treatment_name=treatment)
                
                self.progress.setValue(50)
                
                # Detect subpopulations if requested
                if detect_subpop:
                    subpop_results = self.project.detect_subpopulations(
                        [parameter], 
                        method='kmeans', 
                        treatment_name=treatment
                    )
                    
                    self.progress.setValue(80)
                    
                    # Display subpopulation results
                    self.results_text.append("\n=== Subpopulation Analysis Results ===")
                    if subpop_results.get('multiple_populations_detected', False):
                        self.results_text.append(f"Detected {subpop_results['optimal_n_clusters']} subpopulations")
                        
                        for cluster_name, cluster in subpop_results.get('clusters', {}).items():
                            self.results_text.append(f"\n{cluster_name} (n={cluster['size']})")
                            param_stats = cluster['parameter_statistics'][parameter]
                            self.results_text.append(f"  Mean: {param_stats['mean']:.4f}")
                            self.results_text.append(f"  Std: {param_stats['std']:.4f}")
                            
                            # Count treatments in this cluster
                            if len(cluster.get('treatment_counts', {})) > 1:
                                self.results_text.append("  Treatment distribution:")
                                for treatment_name, count in cluster.get('treatment_counts', {}).items():
                                    percentage = 100 * count / cluster['size']
                                    self.results_text.append(f"    {treatment_name}: {count} ({percentage:.1f}%)")
                        
                        # Plot subpopulations
                        self.figure.clear()
                        ax = self.figure.add_subplot(111)
                        
                        # Get reduced parameters and cluster labels
                        params = subpop_results['reduced_parameters']
                        labels = subpop_results['cluster_labels']
                        
                        # Create scatter plot
                        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
                        
                        for i in range(subpop_results['optimal_n_clusters']):
                            mask = labels == i
                            ax.scatter(params[mask, 0], params[mask, 1], 
                                      c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.7)
                        
                        ax.set_xlabel('Component 1')
                        ax.set_ylabel('Component 2')
                        ax.set_title('Subpopulation Analysis')
                        ax.grid(alpha=0.3)
                        ax.legend()
                        
                        self.canvas.draw()
                    else:
                        self.results_text.append("No distinct subpopulations detected")
                
                # Display pooled results
                self.results_text.append("\n=== Population Analysis Results ===")
                if 'status' in pooled_results:
                    self.results_text.append(f"Status: {pooled_results['status']}")
                else:
                    self.results_text.append(f"Parameter: {pooled_results['parameter']}")
                    self.results_text.append(f"Number of values: {pooled_results['n_values']}")
                    self.results_text.append(f"Number of cells: {pooled_results['n_cells']}")
                    
                    if treatment is None:
                        self.results_text.append(f"Treatments: {', '.join(pooled_results['treatments'])}")
                    
                    self.results_text.append(f"Mean value: {pooled_results['mean']:.4f}")
                    self.results_text.append(f"Std dev: {pooled_results['std']:.4f}")
                    self.results_text.append(f"95% CI: [{pooled_results['ci_95_low']:.4f}, {pooled_results['ci_95_high']:.4f}]")
                    
                    if not detect_subpop or not subpop_results.get('multiple_populations_detected', False):
                        # Plot histogram
                        self.figure.clear()
                        ax = self.figure.add_subplot(111)
                        
                        values = pooled_results['values']
                        
                        ax.hist(values, bins=20, alpha=0.7)
                        ax.axvline(pooled_results['mean'], color='red', linestyle='--', 
                                 label=f'Mean: {pooled_results["mean"]:.4f}')
                        ax.axvline(pooled_results['median'], color='green', linestyle=':', 
                                 label=f'Median: {pooled_results["median"]:.4f}')
                        
                        ax.set_xlabel(parameter.replace('_', ' ').title())
                        ax.set_ylabel('Count')
                        ax.set_title(f'Population Distribution: {parameter}')
                        ax.grid(alpha=0.3)
                        ax.legend()
                        
                        self.canvas.draw()
                
                # Compare treatments if there are multiple
                if treatment is None and len(self.project.treatments) > 1:
                    comparison_results = self.project.compare_treatments(parameter)
                    
                    self.results_text.append("\n=== Treatment Comparison Results ===")
                    for comparison_name, comparison in comparison_results.get('comparisons', {}).items():
                        if 'status' in comparison:
                            self.results_text.append(f"{comparison_name}: {comparison['status']}")
                            continue
                        
                        p_value = comparison['p_value']
                        significant = comparison['significant']
                        effect_size = comparison['effect_size']
                        
                        self.results_text.append(f"{comparison_name}:")
                        self.results_text.append(f"  p-value: {p_value:.4f} {'*' if significant else ''}")
                        self.results_text.append(f"  Effect size: {effect_size:.4f}")
                        
                        if significant:
                            mean_i = comparison['mean_i']
                            mean_j = comparison['mean_j']
                            
                            if mean_i > mean_j:
                                self.results_text.append(f"  {comparison_name.split(' vs ')[0]} > {comparison_name.split(' vs ')[1]}")
                            else:
                                self.results_text.append(f"  {comparison_name.split(' vs ')[0]} < {comparison_name.split(' vs ')[1]}")
                
                self.progress.setValue(100)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during population analysis: {str(e)}")
            self.progress.setValue(0)
    
    def plot_trajectories(self):
        """Plot trajectories from the current data."""
        if self.tracks_df is None:
            return
        
        try:
            # Clear figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Get unique track IDs
            track_ids = self.tracks_df['track_id'].unique()
            
            # Limit to 50 random tracks if too many
            if len(track_ids) > 50:
                np.random.seed(42)  # For reproducibility
                track_ids = np.random.choice(track_ids, 50, replace=False)
            
            # Create colormap
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i) for i in np.linspace(0, 1, len(track_ids))]
            
            # Plot each track
            for i, track_id in enumerate(track_ids):
                track = self.tracks_df[self.tracks_df['track_id'] == track_id]
                track = track.sort_values('frame')
                
                color = colors[i % len(colors)]
                
                # Plot trajectory
                ax.plot(track['x'], track['y'], '-', color=color, linewidth=1, alpha=0.7)
                
                # Mark start and end
                ax.plot(track['x'].iloc[0], track['y'].iloc[0], 'o', color=color, markersize=4)
                ax.plot(track['x'].iloc[-1], track['y'].iloc[-1], 's', color=color, markersize=4)
            
            # Set labels and title
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_title(f'Trajectories (n={len(track_ids)})')
            
            # Invert y-axis to match image coordinates
            ax.invert_yaxis()
            
            # Equal aspect ratio
            ax.set_aspect('equal')
            
            # Draw
            self.canvas.draw()
            
        except Exception as e:
            self.results_text.append(f"Error plotting trajectories: {str(e)}")
    
    def plot_msd(self, ensemble_df, fit_results):
        """Plot MSD curve with fits."""
        try:
            # Clear figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Get data
            time_lags = ensemble_df['time_lag'].values
            msd_values = ensemble_df['msd'].values
            
            # Plot MSD curve
            ax.loglog(time_lags, msd_values, 'o-', label='MSD')
            
            # Plot fits
            x_fit = np.logspace(np.log10(time_lags[0]), np.log10(time_lags[-1]), 100)
            
            # Simple diffusion
            if 'simple_diffusion' in fit_results:
                D = fit_results['simple_diffusion']['D']
                y_fit = 4 * D * x_fit
                ax.loglog(x_fit, y_fit, '--', label=f'Simple: D={D:.4f} μm²/s')
            
            # Anomalous diffusion
            if 'anomalous_diffusion' in fit_results:
                D = fit_results['anomalous_diffusion']['D']
                alpha = fit_results['anomalous_diffusion']['alpha']
                y_fit = 4 * D * x_fit**alpha
                ax.loglog(x_fit, y_fit, '-.', label=f'Anomalous: α={alpha:.2f}')
            
            # Set labels and title
            ax.set_xlabel('Time Lag (s)')
            ax.set_ylabel('MSD (μm²)')
            ax.set_title('Mean Squared Displacement')
            
            # Add grid and legend
            ax.grid(True, which="both", ls="-", alpha=0.3)
            ax.legend()
            
            # Draw
            self.canvas.draw()
            
        except Exception as e:
            self.results_text.append(f"Error plotting MSD: {str(e)}")
    
    def generate_plot(self):
        """Generate plot based on selected plot type."""
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load a file first.")
            return
        
        plot_type = self.plot_type.currentText()
        
        try:
            # Apply plot style
            plt.style.use(self.plot_style.currentText())
            
            if plot_type == 'Trajectories':
                self.plot_trajectories()
            
            elif plot_type == 'MSD Curve':
                # Run MSD analysis
                pixel_size = self.pixel_size.value()
                frame_interval = self.frame_interval.value()
                
                analyzer = diffusion.DiffusionAnalyzer(
                    pixel_size=pixel_size,
                    frame_interval=frame_interval
                )
                
                # Compute MSD
                msd_df, ensemble_df = analyzer.compute_msd(
                    self.tracks_df,
                    max_lag=20,
                    min_track_length=self.min_track_length.value()
                )
                
                # Fit diffusion models
                fit_results = analyzer.fit_diffusion_models(ensemble_df)
                
                # Plot MSD curve
                self.plot_msd(ensemble_df, fit_results)
            
            elif plot_type == 'Diffusion Coefficient':
                # Compute diffusion coefficients for each track
                pixel_size = self.pixel_size.value()
                frame_interval = self.frame_interval.value()
                
                analyzer = diffusion.DiffusionAnalyzer(
                    pixel_size=pixel_size,
                    frame_interval=frame_interval
                )
                
                # Compute diffusion coefficients
                diffusion_df = analyzer.compute_diffusion_coefficient(
                    self.tracks_df,
                    method='individual',
                    min_track_length=self.min_track_length.value()
                )
                
                # Plot diffusion coefficient distribution
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                # Plot histogram
                ax.hist(diffusion_df['D'], bins=20, alpha=0.7)
                
                # Add mean line
                mean_D = diffusion_df['D'].mean()
                ax.axvline(mean_D, color='red', linestyle='--', 
                         label=f'Mean: {mean_D:.4f} μm²/s')
                
                # Set labels and title
                ax.set_xlabel('Diffusion Coefficient (μm²/s)')
                ax.set_ylabel('Count')
                ax.set_title('Diffusion Coefficient Distribution')
                
                # Add grid and legend
                ax.grid(alpha=0.3)
                ax.legend()
                
                # Draw
                self.canvas.draw()
            
            elif plot_type == 'Jump Distribution':
                # Calculate jump sizes
                pixel_size = self.pixel_size.value()
                
                jumps = []
                
                for track_id, track in self.tracks_df.groupby('track_id'):
                    track = track.sort_values('frame')
                    
                    # Skip if track is too short
                    if len(track) < 2:
                        continue
                    
                    # Calculate jumps
                    dx = np.diff(track['x']) * pixel_size
                    dy = np.diff(track['y']) * pixel_size
                    jump_sizes = np.sqrt(dx**2 + dy**2)
                    
                    jumps.extend(jump_sizes)
                
                # Plot jump distribution
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                # Plot histogram
                ax.hist(jumps, bins=30, alpha=0.7)
                
                # Set labels and title
                ax.set_xlabel('Jump Size (μm)')
                ax.set_ylabel('Count')
                ax.set_title('Jump Size Distribution')
                
                # Add grid
                ax.grid(alpha=0.3)
                
                # Draw
                self.canvas.draw()
            
            elif plot_type == 'Cluster Map':
                # Run clustering analysis
                pixel_size = self.pixel_size.value()
                
                analyzer = clustering.ClusterAnalyzer(pixel_size=pixel_size)
                
                # Perform clustering analysis
                cluster_df, clustered_df = analyzer.analyze_clusters(self.tracks_df)
                
                # Plot clusters
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                # Import visualization function
                from ..visualization.clustering import plot_spatial_clusters
                
                # Plot clusters
                plot_spatial_clusters(self.tracks_df, clustered_df, cluster_df, ax=ax, pixel_size=pixel_size)
                
                # Draw
                self.canvas.draw()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating plot: {str(e)}")
    
    def update_plot_style(self, style_name):
        """Update the plot style."""
        plt.style.use(style_name)
        self.generate_plot()  # Regenerate current plot with new style
    
    def save_plot(self):
        """Save the current plot to a file."""
        if not hasattr(self, 'figure') or self.figure is None:
            QMessageBox.warning(self, "Warning", "No plot to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", 
                                                "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.results_text.append(f"Plot saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving plot: {str(e)}")
