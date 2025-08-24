import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and initial validation of datasets"""
    
    def __init__(self, data_dir: str = "resources"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load a single dataset with validation
        Returns:
            Loaded and validated DataFrame
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        try:
            print(f"\nLoading {filename}...")
            df = pd.read_csv(filepath)
            
            print(f"Successfully loaded {len(df):,} rows × {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            raise
    
    def load_churn_data(self) -> pd.DataFrame:
        """Load and validate churn dataset"""
        print(f"\n" + "-"*50)
        print("LOADING CHURN DATASET")
        print("-"*50)
        
        df = self.load_dataset('churn_orgs.csv')
        
        # Additional validation for churn data
        unique_orgs = df['masked_organisation_id'].nunique()
        total_rows = len(df)
        
        print(f"\nChurn Data Validation:")
        if unique_orgs != total_rows:
            print(f"Organizations: {unique_orgs:,} unique out of {total_rows:,} total rows")
        else:
            print(f"Organizations: {unique_orgs:,} (perfect 1:1 mapping)")
        
        # Check churn status values
        churn_values = df['churn_status'].unique()
        churn_counts = df['churn_status'].value_counts()
        
        print(f"\nChurn Status Distribution:")
        for status, count in churn_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {status}: {count:,} ({percentage:.1f}%)")
        
        self.datasets['churn'] = df
        return df
    
    def load_commercial_data(self) -> pd.DataFrame:
        """Load and validate commercial dataset"""
        print(f"\n" + "-"*50)
        print("LOADING COMMERCIAL DATASET")
        print("-"*50)
        
        df = self.load_dataset('data_commercial.csv')
        
        # Convert date columns
        date_columns = ['report_date', 'create_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        print(f"\nCommercial Data Analysis:")
        
        # Validate date ranges
        if 'report_date' in df.columns:
            date_range = df['report_date'].agg(['min', 'max'])
            print(f"   Date Range: {date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}")

        # Validate organization consistency
        unique_orgs = df['masked_organisation_id'].nunique()
        print(f"Coverage: {unique_orgs:,} unique organizations")

        self.datasets['commercial'] = df
        return df
    
    def load_product_data(self) -> pd.DataFrame:
        """Load and validate product dataset"""
        print(f"\n" + "-"*50)
        print("LOADING PRODUCT DATASET")
        print("-"*50)

        df = self.load_dataset('data_product.csv')
        
        # Convert date columns
        date_columns = [
            'report_date', 'create_date', 'current_period_end_datetime',
            'current_period_start_datetime', 'renewal_date'
        ]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        self.datasets['product'] = df
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets
        
        Returns:
            Dictionary containing all loaded datasets
        """

        # Load individual datasets
        churn_df = self.load_churn_data()
        commercial_df = self.load_commercial_data()
        product_df = self.load_product_data()
        
        
        return {
            'churn': churn_df,
            'commercial': commercial_df,
            'product': product_df
        }

    
    def get_data_summary(self) -> Dict:
        """Get summary statistics for all loaded datasets"""
        summary = {}
        
        for name, df in self.datasets.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'unique_orgs': df['masked_organisation_id'].nunique() if 'masked_organisation_id' in df.columns else 0,
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': int(df.isnull().sum().sum()),
                'duplicate_rows': int(df.duplicated().sum())
            }
        
        return summary
    
    def print_data_summary(self) -> None:
        """Print a formatted summary of all loaded datasets"""
        summary = self.get_data_summary()
        
        print(f"\n" + "="*80)
        print("DATA SUMMARY REPORT")
        print("="*80)   
        # Calculate totals
        total_rows = sum(stats['rows'] for stats in summary.values())
        total_memory = sum(stats['memory_mb'] for stats in summary.values())
        
        print(f"\nOVERALL TOTALS:")
        print(f"   Total Rows:       {total_rows:,}")
        print(f"   Total Memory:     {total_memory:.1f} MB")
        print(f"   Datasets Loaded:  {len(summary)}")