"""
Unit tests for the path resolution functionality in the knowledge_base module.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Import the target functionality
from src.kb import get_project_root, KnowledgeBaseManager, MOCKS_DIR, PROJECT_ROOT


class TestPathResolution(unittest.TestCase):
    """Test suite for path resolution functionality."""
    
    def test_project_root_exists(self):
        """Test that the project root detected is an actual directory."""
        root = get_project_root()
        self.assertTrue(os.path.exists(root), f"Project root '{root}' doesn't exist")
        self.assertTrue(os.path.isdir(root), f"Project root '{root}' is not a directory")
    
    def test_mocks_dir_is_under_project_root(self):
        """Test that the mocks directory is under the project root."""
        root = get_project_root()
        self.assertTrue(str(MOCKS_DIR).startswith(str(root)), 
                       f"MOCKS_DIR '{MOCKS_DIR}' is not under project root '{root}'")
    
    def test_mocks_dir_name(self):
        """Test that the mocks directory has the correct name."""
        self.assertTrue(MOCKS_DIR.endswith("__mocks__"), 
                       f"MOCKS_DIR '{MOCKS_DIR}' doesn't end with '__mocks__'")
    
    def test_strategy1_project_root(self):
        """Test Strategy 1: Finding the root by key project directories."""
        # Create a mock implementation of exists that knows about __mocks__
        def mock_exists(self):
            # This will be called on a Path object like: (potential_root / "__mocks__")
            return "__mocks__" in str(self)
        
        # Patch the exists method on the Path class
        with patch('pathlib.Path.exists', mock_exists), \
             patch('os.path.abspath', return_value='/fake/path/src/knowledge_base.py'):
            
            # Run the function with mocks in place
            root = get_project_root()
            
            # Check that it used the first strategy correctly
            self.assertEqual(str(root), '/fake/path', 
                            "Strategy 1 didn't return the correct project root")
    
    def test_strategy2_cwd(self):
        """Test Strategy 2: Using the current working directory."""
        # Create a mock for the Path.exists method
        original_exists = Path.exists
        
        def mock_exists(self):
            path_str = str(self)
            # Make strategy 1 fail
            if '/fake/path/__mocks__' in path_str:
                return False
            # But make strategy 2 succeed
            if '/current/work/dir/__mocks__' in path_str or '/current/work/dir/src' in path_str:
                return True
            # For any other paths, use the original exists method
            return original_exists(self)
        
        # Apply the mocks
        with patch('pathlib.Path.exists', mock_exists), \
             patch('os.path.abspath', return_value='/fake/path/src/knowledge_base.py'), \
             patch('pathlib.Path.cwd', return_value=Path('/current/work/dir')):
                
            # Run the function with mocks in place
            root = get_project_root()
            
            # Check that it used the second strategy correctly
            self.assertEqual(str(root), '/current/work/dir', 
                            "Strategy 2 didn't return the current working directory")
    
    def test_strategy3_parent_cwd(self):
        """Test Strategy 3: Using the parent of the current working directory."""
        # Create a mock for the Path.exists method
        original_exists = Path.exists
        
        def mock_exists(self):
            path_str = str(self)
            # Make strategies 1 and 2 fail
            if '/some/other/path/__mocks__' in path_str or '/unrelated/dir/__mocks__' in path_str:
                return False
            if '/unrelated/dir/src' in path_str:
                return False
            # But make strategy 3 succeed
            if '/parent/dir/__mocks__' in path_str or '/parent/dir/src' in path_str:
                return True
            # For any other paths, use the original exists method
            return original_exists(self)
        
        # Apply the mocks
        with patch('pathlib.Path.exists', mock_exists), \
             patch('os.path.abspath', return_value='/some/other/path/src/knowledge_base.py'), \
             patch('pathlib.Path.cwd', return_value=Path('/parent/dir/subdir')):
                
            # Run the function with mocks in place
            root = get_project_root()
            
            # The function should return the parent of cwd
            self.assertEqual(str(root), '/parent/dir', 
                            "Strategy 3 didn't return the parent of current working directory")
    
    def test_strategy4_sys_path(self):
        """Test Strategy 4: Looking in sys.path."""
        # Create a mock for the Path.exists method
        original_exists = Path.exists
        
        def mock_exists(self):
            path_str = str(self)
            # Make strategies 1, 2, and 3 fail
            if any(x in path_str for x in ['/some/other/path/__mocks__', '/unrelated/dir/__mocks__']):
                return False
            if any(x in path_str for x in ['/unrelated/dir/src', '/unrelated/dir/parent/__mocks__']):
                return False
            # But make strategy 4 succeed for the second path in sys.path
            if '/python/lib/site-packages/__mocks__' in path_str or '/python/lib/site-packages/src' in path_str:
                return True
            # For any other paths, use the original exists method
            return original_exists(self)
        
        # Apply the mocks
        with patch('pathlib.Path.exists', mock_exists), \
             patch('os.path.abspath', return_value='/some/other/path/src/knowledge_base.py'), \
             patch('pathlib.Path.cwd', return_value=Path('/unrelated/dir')), \
             patch.object(sys, 'path', ['/wrong/path', '/python/lib/site-packages']):
                
            # Run the function with mocks in place
            root = get_project_root()
            
            # The function should use the right path from sys.path
            self.assertEqual(str(root), '/python/lib/site-packages', 
                            "Strategy 4 didn't return a path from sys.path")
    
    def test_fallback_strategy(self):
        """Test the fallback strategy."""
        with patch('pathlib.Path.exists', return_value=False):  # All other strategies fail
            with patch('os.path.abspath', return_value='/fallback/path/src/knowledge_base.py'):
                root = get_project_root()
                self.assertEqual(str(root), '/fallback/path', 
                                "Fallback strategy didn't return two levels up from file")


class TestKnowledgeBaseManager:
    """Test suite for the KnowledgeBaseManager path resolution."""
    
    def test_default_mocks_dir(self):
        """Test that the default mocks_dir is set correctly."""
        kb = KnowledgeBaseManager()
        assert kb.mocks_dir == MOCKS_DIR, f"Default mocks_dir '{kb.mocks_dir}' != '{MOCKS_DIR}'"
    
    def test_custom_mocks_dir(self):
        """Test that a custom mocks_dir is respected."""
        custom_dir = "/custom/path"
        kb = KnowledgeBaseManager(mocks_dir=custom_dir)
        assert kb.mocks_dir == custom_dir, f"Custom mocks_dir '{kb.mocks_dir}' != '{custom_dir}'"
    
    def test_index_path_is_absolute(self):
        """Test that the index_path is an absolute path."""
        kb = KnowledgeBaseManager()
        assert os.path.isabs(kb.index_path), f"index_path '{kb.index_path}' is not absolute"
    
    def test_documents_pickle_path_is_under_index_path(self):
        """Test that documents_pickle_path is under index_path."""
        kb = KnowledgeBaseManager()
        assert kb.documents_pickle_path.startswith(kb.index_path), \
               f"documents_pickle_path '{kb.documents_pickle_path}' is not under index_path '{kb.index_path}'"


if __name__ == '__main__':
    unittest.main() 