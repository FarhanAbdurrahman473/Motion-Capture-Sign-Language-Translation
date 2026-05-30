"""
Motion Capture Module

This module handles video capture from webcam and motion tracking
using MediaPipe for landmark extraction of hand and body movements.

For data collection, use: python -m src.collect_data
For validation, use: python -m src.validate_dataset
"""

from src.collect_data import DataCollector

__all__ = ['DataCollector']