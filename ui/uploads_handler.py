"""
Upload Handler Module
Manages CSV file uploads, validation, and initial preprocessing.
"""

import streamlit as st
import pandas as pd
from typing import Tuple, Optional
import os


class UploadHandler:
    """
    Handles file uploads, validation, and session state management.
    
    Responsibilities:
    - Validate CSV format
    - Check file size
    - Load CSV into DataFrame
    - Store in Streamlit session state
    - Display upload UI
    """
    
    MAX_FILE_SIZE_MB = 100  # Maximum 100MB file size
    
    @staticmethod
    def validate_csv(file) -> Tuple[bool, str]:
        """
        Validate uploaded CSV file.
        
        Args:
            file: Uploaded file object from Streamlit
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if file is None:
            return False, "No file selected"
        
        # Check file extension
        if not file.name.endswith('.csv'):
            return False, "File must be a CSV file"
        
        # Check file size
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > UploadHandler.MAX_FILE_SIZE_MB:
            return False, f"File size exceeds {UploadHandler.MAX_FILE_SIZE_MB}MB limit"
        
        return True, ""
    
    @staticmethod
    def load_csv(file) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load CSV file into pandas DataFrame.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Tuple[DataFrame, error_message]: (dataframe or None, error_message)
        """
        try:
            df = pd.read_csv(file)
            
            # Validate dataframe
            if df.empty:
                return None, "CSV file is empty"
            
            if df.shape[1] < 2:
                return None, "CSV must have at least 2 columns"
            
            return df, ""
        
        except pd.errors.ParserError:
            return None, "Failed to parse CSV. Check formatting."
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    @staticmethod
    def render_upload_section() -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Render the upload section in Streamlit UI.
        
        Returns:
            Tuple[uploaded, dataframe, target_column]:
            - uploaded (bool): Whether file was successfully uploaded
            - dataframe (pd.DataFrame): Loaded dataset or None
            - target_column (str): Selected target column or None
        """
        st.subheader("📂 Upload Dataset")
        
        # Upload widget
        uploaded_file = st.file_uploader(
            "Upload your CSV dataset",
            type=["csv"],
            help="Maximum file size: 100MB"
        )
        
        if uploaded_file is None:
            st.info("👆 Upload a CSV file to get started")
            return False, None, None
        
        # Validate file
        is_valid, error_msg = UploadHandler.validate_csv(uploaded_file)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return False, None, None
        
        # Load CSV
        df, error_msg = UploadHandler.load_csv(uploaded_file)
        if df is None:
            st.error(f"❌ {error_msg}")
            return False, None, None
        
        # Store in session state
        st.session_state.dataframe = df
        st.session_state.original_filename = uploaded_file.name
        
        st.success("✅ Dataset uploaded successfully!")
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        
        # Target column selection
        st.markdown("---")
        st.subheader("🎯 Select Target Column")
        st.caption("Choose the column you want to predict or analyze")
        
        target_column = st.selectbox(
            "Target Column",
            options=df.columns.tolist(),
            key="target_column_select",
            help="This is the column your models will predict"
        )
        
        if target_column:
            st.session_state.target_column = target_column
            st.info(f"✓ Target column set to: **{target_column}**")
            return True, df, target_column
        
        return False, df, None
    
    @staticmethod
    def get_dataset_from_session() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Retrieve dataset and target column from session state.
        
        Returns:
            Tuple[dataframe, target_column]: Stored values or (None, None)
        """
        df = st.session_state.get("dataframe", None)
        target = st.session_state.get("target_column", None)
        return df, target
    
    @staticmethod
    def clear_dataset():
        """Clear dataset from session state."""
        if "dataframe" in st.session_state:
            del st.session_state.dataframe
        if "target_column" in st.session_state:
            del st.session_state.target_column
