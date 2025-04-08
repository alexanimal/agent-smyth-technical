"""
Configuration file for pytest.
This file is automatically loaded by pytest and can be used to define fixtures and setup code.
"""
import os
import sys

# Add the project root directory to the Python path
# This ensures that imports from the project root work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) 