"""
Microenvironment analysis module for SPT Analysis.

This module provides tools for segmenting, analyzing, and classifying subcellular
compartments and relating them to particle tracking data.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
import skimage.segmentation as segmentation
from skimage.measure import regionprops, label
from skimage import filters
import cv2
import os
import json
import uuid
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class CompartmentDefinition:
    """
    Definition of a compartment based on channel intensity criteria.
    
    This class encapsulates the rules for identifying a compartment based on
    intensity thresholds from one or more imaging channels.
    
    Parameters
    ----------
    name : str
        Name of the compartment.
    description : str, optional
        Description of the compartment, by default None.
    rules : list, optional
        List of rules defining the compartment, by default None.
    color : tuple, optional
        RGB color tuple for visualization, by default None.
    """
    
    def __init__(self, name, description=None, rules=None, color=None):
        self.name = name
        self.description = description or ""
        self.rules = rules or []
        self.color = color or (0.5, 0.5, 0.5)  # Default gray color
        self.id = str(uuid.uuid4())
        
    def add_rule(self, channel_name, min_percentile=None, max_percentile=None, 
                 min_absolute=None, max_absolute=None):
        """
        Add a rule for this compartment.
        
        Parameters
        ----------
        channel_name : str
            Name of the channel this rule applies to.
        min_percentile : float, optional
            Minimum intensity percentile (0-1), by default None.
        max_percentile : float, optional
            Maximum intensity percentile (0-1), by default None.
        min_absolute : float, optional
            Minimum absolute intensity value, by default None.
        max_absolute : float, optional
            Maximum absolute intensity value, by default None.
            
        Returns
        -------
        dict
            The newly added rule.
        """
        rule = {
            'channel': channel_name,
            'min_percentile': min_percentile,
            'max_percentile': max_percentile,
            'min_absolute': min_absolute,
            'max_absolute': max_absolute
        }
        self.rules.append(rule)
        return rule
    
    def to_dict(self):
        """
        Convert compartment definition to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the compartment definition.
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'rules': self.rules.copy(),
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create compartment definition from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary containing compartment definition.
            
        Returns
        -------
        CompartmentDefinition
            Created compartment definition.
        """
        definition = cls(
            name=data['name'],
            description=data.get('description', ""),
            rules=data.get('rules', []),
            color=data.get('color', (0.5, 0.5, 0.5))
        )
        definition.id = data.get('id', str(uuid.uuid4()))
        return definition
    
    def generate_mask(self, channel_images, thresholds):
        """
        Generate binary mask for this compartment.
        
        Parameters
        ----------
        channel_images : dict
            Dictionary mapping channel names to image arrays.
        thresholds : dict
            Dictionary mapping channel names to threshold values.
            
        Returns
        -------
        numpy.ndarray
            Binary mask for the compartment.
        """
        if not self.rules or not channel_images:
            return None
        
        # Use the first available channel image to determine dimensions.
        first_channel = next(iter(channel_images.values()))
        mask = np.ones_like(first_channel, dtype=bool)
        
        # Sequentially apply each rule.
        for rule in self.rules:
            channel = rule['channel']
            
            # Skip if the specified channel is not available in the provided images.
            if channel not in channel_images:
                continue
            
            image = channel_images[channel]
            
            # Apply minimum percentile threshold.
            if rule['min_percentile'] is not None:
                if channel in thresholds and f"p{rule['min_percentile']:.2f}" in thresholds[channel]:
                    threshold = thresholds[channel][f"p{rule['min_percentile']:.2f}"]
                else:
                    threshold = np.percentile(image[image > 0], rule['min_percentile'] * 100)
                mask = mask & (image >= threshold)
            
            # Apply maximum percentile threshold.
            if rule['max_percentile'] is not None:
                if channel in thresholds and f"p{rule['max_percentile']:.2f}" in thresholds[channel]:
                    threshold = thresholds[channel][f"p{rule['max_percentile']:.2f}"]
                else:
                    threshold = np.percentile(image[image > 0], rule['max_percentile'] * 100)
                mask = mask & (image <= threshold)
            
            # Apply minimum absolute threshold.
            if rule['min_absolute'] is not None:
                mask = mask & (image >= rule['min_absolute'])
            
            # Apply maximum absolute threshold.
            if rule['max_absolute'] is not None:
                mask = mask & (image <= rule['max_absolute'])
        
        return mask


class CompartmentConfigManager:
    """
    Manager for compartment definitions and configurations.
    
    This class provides methods for creating, managing, and applying
    user-defined compartment definitions. It supports operations such as
    updating, deleting, and persisting configurations.
    """
    
    def __init__(self):
        self.compartment_definitions = {}
        self.current_config_name = "Default"
        self.configs = {"Default": self._create_default_definitions()}
        
    def _create_default_definitions(self):
        """
        Create default nuclear compartment definitions.
        
        Returns
        -------
        dict
            Dictionary of default compartment definitions.
        """
        # Heterochromatin: DNA high, splicing factor low.
        heterochromatin = CompartmentDefinition(
            name="Heterochromatin",
            description="DNA-dense regions with low splicing factor",
            color=(0, 0, 0.8)  # Blue
        )
        heterochromatin.add_rule("DAPI/DNA", min_percentile=0.85)
        heterochromatin.add_rule("Splicing Factor", max_percentile=0.3)
        
        # Euchromatin: DNA moderate, splicing factor low.
        euchromatin = CompartmentDefinition(
            name="Euchromatin",
            description="Moderate DNA density with low splicing factor",
            color=(0, 0.5, 1.0)  # Light blue
        )
        euchromatin.add_rule("DAPI/DNA", min_percentile=0.6, max_percentile=0.85)
        euchromatin.add_rule("Splicing Factor", max_percentile=0.3)
        
        # Nucleoplasm: DNA low, splicing factor moderate.
        nucleoplasm = CompartmentDefinition(
            name="Nucleoplasm",
            description="Low DNA density with moderate splicing factor",
            color=(0.7, 0.7, 0.7)  # Gray
        )
        nucleoplasm.add_rule("DAPI/DNA", max_percentile=0.3)
        nucleoplasm.add_rule("Splicing Factor", min_percentile=0.3, max_percentile=0.7)
        
        # Nuclear speckles: splicing factor high.
        speckles = CompartmentDefinition(
            name="Nuclear Speckles",
            description="Regions with high splicing factor concentration",
            color=(0, 0.8, 0)  # Green
        )
        speckles.add_rule("Splicing Factor", min_percentile=0.9)
        
        # Nucleolus: DNA very low, splicing factor very low.
        nucleolus = CompartmentDefinition(
            name="Nucleolus",
            description="Regions with very low DNA and splicing factor",
            color=(1.0, 0.8, 0)  # Orange
        )
        nucleolus.add_rule("DAPI/DNA", max_percentile=0.1)
        nucleolus.add_rule("Splicing Factor", max_percentile=0.1)
        
        return {
            heterochromatin.id: heterochromatin,
            euchromatin.id: euchromatin,
            nucleoplasm.id: nucleoplasm,
            speckles.id: speckles,
            nucleolus.id: nucleolus
        }
    
    def create_definition(self, name, description=None, rules=None, color=None):
        """
        Create a new compartment definition.
        
        Parameters
        ----------
        name : str
            Name of the compartment.
        description : str, optional
            Description of the compartment, by default None.
        rules : list, optional
            List of rules defining the compartment, by default None.
        color : tuple, optional
            RGB color tuple for visualization, by default None.
            
        Returns
        -------
        CompartmentDefinition
            The newly created compartment definition.
        """
        definition = CompartmentDefinition(name, description, rules, color)
        self.configs[self.current_config_name][definition.id] = definition
        return definition
    
    def get_definition(self, definition_id):
        """
        Get a compartment definition by its ID.
        
        Parameters
        ----------
        definition_id : str
            ID of the compartment definition.
            
        Returns
        -------
        CompartmentDefinition
            The requested compartment definition.
            
        Raises
        ------
        KeyError
            If the definition does not exist.
        """
        if definition_id not in self.configs[self.current_config_name]:
            raise KeyError(f"Compartment definition '{definition_id}' does not exist.")
        
        return self.configs[self.current_config_name][definition_id]
    
    def update_definition(self, definition_id, name=None, description=None, rules=None, color=None):
        """
        Update an existing compartment definition.
        
        Parameters
        ----------
        definition_id : str
            ID of the compartment definition.
        name : str, optional
            New name of the compartment, by default None.
        description : str, optional
            New description of the compartment, by default None.
        rules : list, optional
            New rules for the compartment, by default None.
        color : tuple, optional
            New RGB color tuple for visualization, by default None.
            
        Returns
        -------
        CompartmentDefinition
            The updated compartment definition.
            
        Raises
        ------
        KeyError
            If the specified compartment definition does not exist.
        """
        if definition_id not in self.configs[self.current_config_name]:
            raise KeyError(f"Compartment definition '{definition_id}' does not exist.")
        
        definition = self.configs[self.current_config_name][definition_id]
        if name is not None:
            definition.name = name
        if description is not None:
            definition.description = description
        if rules is not None:
            definition.rules = rules
        if color is not None:
            definition.color = color
        return definition
    
    def delete_definition(self, definition_id):
        """
        Delete a compartment definition from the current configuration.
        
        Parameters
        ----------
        definition_id : str
            ID of the compartment definition to delete.
            
        Raises
        ------
        KeyError
            If the compartment definition does not exist.
        """
        if definition_id not in self.configs[self.current_config_name]:
            raise KeyError(f"Compartment definition '{definition_id}' does not exist.")
        del self.configs[self.current_config_name][definition_id]
        logger.info(f"Deleted compartment definition with ID {definition_id}.")
    
    def export_config(self, file_path: str):
        """
        Export all configurations to a JSON file.
        
        Parameters
        ----------
        file_path : str
            File path where the configuration will be saved.
        """
        config_data = {
            config_name: {def_id: def_obj.to_dict() for def_id, def_obj in config.items()}
            for config_name, config in self.configs.items()
        }
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Configuration saved to {file_path}.")
    
    def import_config(self, file_path: str):
        """
        Import configurations from a JSON file.
        
        Parameters
        ----------
        file_path : str
            File path from which the configuration will be loaded.
            
        Raises
        ------
        FileNotFoundError
            If the specified configuration file does not exist.
        """
        if not os.path.exists(file_path):
            logger.error(f"Configuration file {file_path} does not exist.")
            raise FileNotFoundError(f"Configuration file {file_path} does not exist.")
        
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        for config_name, definitions in config_data.items():
            self.configs[config_name] = {
                def_id: CompartmentDefinition.from_dict(def_data)
                for def_id, def_data in definitions.items()
            }
        self.current_config_name = next(iter(self.configs))
        logger.info(f"Configuration imported from {file_path}.")
    
    def set_current_config(self, config_name: str):
        """
        Change the current active configuration.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration to set as current.
            
        Raises
        ------
        KeyError
            If the specified configuration does not exist.
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration '{config_name}' does not exist.")
        self.current_config_name = config_name
        logger.info(f"Current configuration set to '{config_name}'.")


def analyze_compartments(compartment_masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze segmented compartment masks and compute quantitative metrics.
    
    This function labels regions within each compartment mask and computes summary
    statistics such as the number of regions, mean region area, and total area.
    
    Parameters
    ----------
    compartment_masks : dict
        Dictionary mapping compartment names to binary mask arrays.
        
    Returns
    -------
    dict
        A dictionary where each key is a compartment name and the value is a dictionary
        containing analysis metrics.
    """
    analysis_results = {}
    for comp_name, mask in compartment_masks.items():
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)
        if regions:
            areas = [region.area for region in regions]
            analysis_results[comp_name] = {
                "num_regions": len(regions),
                "mean_area": np.mean(areas),
                "total_area": np.sum(areas)
            }
        else:
            analysis_results[comp_name] = {
                "num_regions": 0,
                "mean_area": 0,
                "total_area": 0
            }
    return analysis_results


if __name__ == '__main__':
    # This section demonstrates basic usage of the module.
    # Example images would be loaded from appropriate imaging data.
    # Here, dummy images and thresholds are constructed for illustrative purposes.
    
    # Create dummy channel images.
    image_shape = (512, 512)
    dapi_image = np.random.random(image_shape)
    sf_image = np.random.random(image_shape)
    channel_images = {"DAPI/DNA": dapi_image, "Splicing Factor": sf_image}
    
    # Generate thresholds for each channel.
    thresholds = {
        "DAPI/DNA": {
            "p0.60": np.percentile(dapi_image[dapi_image > 0], 60),
            "p0.85": np.percentile(dapi_image[dapi_image > 0], 85),
            "p0.30": np.percentile(dapi_image[dapi_image > 0], 30)
        },
        "Splicing Factor": {
            "p0.90": np.percentile(sf_image[sf_image > 0], 90),
            "p0.30": np.percentile(sf_image[sf_image > 0], 30),
            "p0.70": np.percentile(sf_image[sf_image > 0], 70),
            "p0.10": np.percentile(sf_image[sf_image > 0], 10)
        }
    }
    
    # Initialize the configuration manager and retrieve default compartments.
    config_manager = CompartmentConfigManager()
    compartments = config_manager.configs[config_manager.current_config_name]
    
    # Generate a binary mask for each compartment based on its defined rules.
    compartment_masks = {}
    for comp_def in compartments.values():
        mask = comp_def.generate_mask(channel_images, thresholds)
        if mask is not None:
            compartment_masks[comp_def.name] = mask
    
    # Analyze the compartment masks.
    analysis = analyze_compartments(compartment_masks)
    print("Compartment Analysis Results:")
    for comp, metrics in analysis.items():
        print(f"{comp}: {metrics}")
