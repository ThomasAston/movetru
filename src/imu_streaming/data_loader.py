"""Data loading and validation utilities for IMU data."""

import polars as pl
from pathlib import Path
from typing import Tuple, Optional


class IMUDataLoader:
    """Handles loading and validation of IMU data files."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing IMU parquet files
        """
        self.data_dir = data_dir
    
    def get_available_players(self) -> list[str]:
        """
        Extract player IDs from available LF (left foot) files.
        
        Returns:
            List of player ID strings
        """
        files = sorted(self.data_dir.glob("*_LF.parquet"))
        return [f.stem.replace("_LF", "") for f in files]
    
    def load_player_data(self, player_id: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load left and right foot data for a specific player.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Tuple of (left_foot_df, right_foot_df)
            
        Raises:
            FileNotFoundError: If data files don't exist for the player
        """
        lf_path = self.data_dir / f"{player_id}_LF.parquet"
        rf_path = self.data_dir / f"{player_id}_RF.parquet"
        
        if not lf_path.exists() or not rf_path.exists():
            raise FileNotFoundError(f"Data files not found for player {player_id}")
        
        return pl.read_parquet(lf_path), pl.read_parquet(rf_path)
    
    def time_to_sample_index(self, df: pl.DataFrame, start_time: float) -> int:
        """
        Convert time in seconds to sample index.
        
        Args:
            df: DataFrame with 'Time' column
            start_time: Time in seconds
            
        Returns:
            Sample index corresponding to the time
        """
        for idx in range(len(df)):
            if df['Time'][idx] >= start_time:
                return idx
        return len(df)  # Return last index if time exceeds data
    
    def validate_start_position(
        self, 
        df_lf: pl.DataFrame, 
        df_rf: pl.DataFrame, 
        start_index: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that start position is within data bounds.
        
        Args:
            df_lf: Left foot DataFrame
            df_rf: Right foot DataFrame
            start_index: Starting sample index
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        min_len = min(len(df_lf), len(df_rf))
        if start_index >= min_len:
            max_time = df_lf['Time'][-1]
            return False, f"Start time is beyond available data (max time: {max_time:.2f}s)"
        return True, None
    
    def get_file_paths(self, player_id: str) -> Tuple[Path, Path]:
        """
        Get file paths for a player's left and right foot data.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Tuple of (lf_path, rf_path)
        """
        lf_path = self.data_dir / f"{player_id}_LF.parquet"
        rf_path = self.data_dir / f"{player_id}_RF.parquet"
        return lf_path, rf_path
