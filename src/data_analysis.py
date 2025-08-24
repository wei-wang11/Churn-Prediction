import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class InitialDataAnalysis:
    """
    Simplified Churn Data Analysis
    
    Purpose:
    1. Analyze industry and country distributions
    2. Create two datasets: merged_df (time-series) and org_features (aggregated)
    3. Generate basic statistics and visualizations
    """
    
    def __init__(self, datasets, output_dir="analysis_output", churn_status_column="churn_status"):
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Churn definition
        self.churn_status_column = churn_status_column
        self.churn_status_value = "Churned"
        
        # Results storage
        self.merged_df = None
        self.org_features = None
        
        plt.style.use('default')
        sns.set_palette("Set2")

    def create_datasets(self):
        """Create the two main datasets: merged_df and org_features"""
        print("="*60)
        print("CREATING DATASETS")
        print("="*60)
        
        # Load and merge datasets
        commercial_df = self.datasets['commercial']
        product_df = self.datasets['product']
        
        print(f"Commercial data: {len(commercial_df):,} rows")
        print(f"Product data: {len(product_df):,} rows")
        overlap = commercial_df.columns.intersection(product_df.columns).difference(['masked_organisation_id', 'report_date'])
        product_clean = product_df.drop(columns=overlap)
        # Create merged_df (time-series data)
        merged_df = commercial_df.merge(
            product_clean, 
            on=['masked_organisation_id', 'report_date'], 
            how='left'
        )
        
        # Clean merged_df
        noisy_columns = ['employee_count_band_description', 'total_seat_count_buckets']
        merged_df = merged_df.drop(columns=noisy_columns, errors='ignore')
        
        # Remove empty columns
        empty_columns = merged_df.columns[merged_df.isnull().all()]
        merged_df = merged_df.drop(columns=empty_columns, errors='ignore')
        
        print(f"Merged dataset: {len(merged_df):,} rows × {len(merged_df.columns)} columns")
        
        # Define churn labels
        churn_labels = self._define_churn_labels(merged_df)
        
        # Add churn to merged_df
        merged_df['is_churned'] = merged_df['masked_organisation_id'].map(churn_labels)
        merged_df = merged_df.dropna(subset=['is_churned'])
        merged_df = merged_df.sort_values(['masked_organisation_id', 'report_date']).reset_index(drop=True)
        
        # Create org_features (aggregated data)
        org_features = self._create_org_features(merged_df, churn_labels)
        
        self.merged_df = merged_df
        self.org_features = org_features
        
        print(f"\nDatasets created:")
        print(f"  merged_df: {len(merged_df):,} rows × {len(merged_df.columns)} columns")
        print(f"  org_features: {len(org_features):,} rows × {len(org_features.columns)} columns")
        print(f"  Churn rate: {org_features['is_churned'].mean():.1%}")
        
        return merged_df, org_features

    def _define_churn_labels(self, merged_df):
        """Define churn labels based on final organization status"""
        print("\nDefining churn labels...")
        
        churn_labels = {}
        
        if (self.churn_status_column and 
            self.churn_status_column in merged_df.columns):
            
            # Get latest status for each organization
            latest_status = (merged_df
                           .sort_values('report_date')
                           .groupby('masked_organisation_id')[self.churn_status_column]
                           .last())
            
            for org_id, status in latest_status.items():
                churn_labels[org_id] = (status == self.churn_status_value)
            
            print(f"Status distribution: {latest_status.value_counts().to_dict()}")
        
        elif 'churn' in self.datasets:
            churn_df = self.datasets['churn']
            churned_orgs = set(churn_df['masked_organisation_id'])
            all_orgs = set(merged_df['masked_organisation_id'])
            
            for org_id in all_orgs:
                churn_labels[org_id] = org_id in churned_orgs
        
        else:
            raise ValueError("No churn definition available")
        
        total_orgs = len(churn_labels)
        churned_count = sum(churn_labels.values())
        
        print(f"Churn labels: {total_orgs:,} organizations, {churned_count:,} churned ({churned_count/total_orgs:.1%})")
        
        return churn_labels

    def _create_org_features(self, merged_df, churn_labels):
        """Create organization-level features by aggregating time-series data"""
        print("\nCreating organization features...")
        
        # Identify column types
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'masked_organisation_id']
        
        categorical_cols = merged_df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['masked_organisation_id', 'report_date']]
        
        # Aggregation strategy
        agg_dict = {}
        
        # Numeric: mean, std, min, max
        for col in numeric_cols:
            agg_dict[col] = ['mean', 'std', 'min', 'max']
        
        # Categorical: mode
        for col in categorical_cols:
            agg_dict[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        
        # Special handling for report_date
        agg_dict['report_date'] = ['count', 'min', 'max']
        
        # Aggregate
        org_grouped = merged_df.groupby('masked_organisation_id').agg(agg_dict)
        
        # Flatten column names
        org_grouped.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                              for col in org_grouped.columns.values]
        org_features = org_grouped.reset_index()
        
        # Clean column names
        org_features.columns = [col.replace('<lambda>', 'mode') for col in org_features.columns]
        
        # Add derived features
        if 'report_date_count' in org_features.columns:
            org_features['reporting_frequency'] = org_features['report_date_count']
        
        # Add churn labels
        org_features['is_churned'] = org_features['masked_organisation_id'].map(churn_labels)
        org_features = org_features.dropna(subset=['is_churned'])
        
        print(f"Organization features: {len(org_features):,} organizations × {len(org_features.columns)} features")
        
        return org_features

    def analyze_basic_statistics(self):
        """Analyze basic statistics for both datasets"""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        if self.merged_df is None or self.org_features is None:
            raise ValueError("Must create datasets first")
        
        # Merged dataset statistics
        print("\n1. Merged Dataset (Time-Series) Statistics:")
        print("-" * 50)
        print(f"Total records: {len(self.merged_df):,}")
        print(f"Unique organizations: {self.merged_df['masked_organisation_id'].nunique():,}")
        print(f"Date range: {self.merged_df['report_date'].min()} to {self.merged_df['report_date'].max()}")
        print(f"Average records per organization: {len(self.merged_df) / self.merged_df['masked_organisation_id'].nunique():.1f}")
        
        # Churn distribution in time-series data
        churn_dist = self.merged_df['is_churned'].value_counts()
        print(f"\nChurn distribution in time-series data:")
        print(f"  Active: {churn_dist.get(False, 0):,} records")
        print(f"  Churned: {churn_dist.get(True, 0):,} records")
        
        # Organization features statistics
        print("\n2. Organization Features (Aggregated) Statistics:")
        print("-" * 50)
        print(f"Total organizations: {len(self.org_features):,}")
        churn_rate = self.org_features['is_churned'].mean()
        print(f"Overall churn rate: {churn_rate:.1%}")
        print(f"Active organizations: {(~self.org_features['is_churned']).sum():,}")
        print(f"Churned organizations: {self.org_features['is_churned'].sum():,}")
        
        # Column types breakdown
        numeric_cols = self.org_features.select_dtypes(include=[np.number]).columns
        categorical_cols = self.org_features.select_dtypes(include=['object']).columns
        
        print(f"\nFeature breakdown:")
        print(f"  Numeric features: {len(numeric_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        print(f"  Total features: {len(self.org_features.columns)}")

    def analyze_industry_country(self):
        """Analyze industry and country distributions by churn"""
        print("\n" + "="*60)
        print("INDUSTRY AND COUNTRY ANALYSIS")
        print("="*60)
        
        if self.org_features is None:
            raise ValueError("Must create datasets first")
        
        # Find industry and country columns
        industry_col = None
        country_col = None
        
        for col in self.org_features.columns:
            if 'industry' in col.lower():
                industry_col = col
                break
        
        for col in self.org_features.columns:
            if 'country' in col.lower():
                country_col = col
                break
        
        # Analyze Industry
        if industry_col:
            print(f"\nIndustry Analysis (using {industry_col}):")
            industry_stats = self._analyze_categorical_churn(self.org_features, industry_col, 'Industry')
            self._visualize_categorical_churn(industry_stats, 'Industry', '#e74c3c')
        else:
            print("\nNo industry column found in dataset")
        
        # Analyze Country
        if country_col:
            print(f"\nCountry Analysis (using {country_col}):")
            country_stats = self._analyze_categorical_churn(self.org_features, country_col, 'Country')
            self._visualize_categorical_churn(country_stats, 'Country', '#f39c12')
        else:
            print("\nNo country column found in dataset")

    def _analyze_categorical_churn(self, df, column, category_name):
        """Analyze churn by categorical variable"""
        stats = df.groupby(column)['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
        stats.columns = [category_name, 'Total_Customers', 'Churned_Count', 'Churn_Rate']
        stats = stats[stats['Total_Customers'] >= 3]  # Filter small categories
        stats = stats.sort_values('Churn_Rate', ascending=False)
        
        print(f"\nTop 10 {category_name} by Churn Rate:")
        print("-" * 70)
        for _, row in stats.head(10).iterrows():
            name = str(row[category_name])[:30] if pd.notna(row[category_name]) else 'Unknown'
            print(f"{name:<30} | Total: {row['Total_Customers']:>4} | "
                 f"Churned: {row['Churned_Count']:>3} | Rate: {row['Churn_Rate']*100:>5.1f}%")
        
        return stats

    def _visualize_categorical_churn(self, stats, category_name, color):
        """Create visualization for categorical churn analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{category_name} Churn Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Churn Rate
        top_15 = stats.head(15)
        ax1.barh(range(len(top_15)), top_15['Churn_Rate'] * 100, color=color, alpha=0.8)
        ax1.set_yticks(range(len(top_15)))
        ax1.set_yticklabels([str(name)[:25] for name in top_15[category_name]], fontsize=10)
        ax1.set_xlabel('Churn Rate (%)', fontweight='bold')
        ax1.set_title(f'Churn Rate by {category_name} (Top 15)', fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_15['Churn_Rate'] * 100):
            ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)
        
        # Plot 2: Customer Volume (stacked)
        top_15_volume = stats.sort_values('Total_Customers', ascending=False).head(15)
        active_customers = top_15_volume['Total_Customers'] - top_15_volume['Churned_Count']
        
        ax2.barh(range(len(top_15_volume)), active_customers, color='#27ae60', alpha=0.8, label='Active')
        ax2.barh(range(len(top_15_volume)), top_15_volume['Churned_Count'], 
                left=active_customers, color='#e74c3c', alpha=0.8, label='Churned')
        
        ax2.set_yticks(range(len(top_15_volume)))
        ax2.set_yticklabels([str(name)[:25] for name in top_15_volume[category_name]], fontsize=10)
        ax2.set_xlabel('Number of Customers', fontweight='bold')
        ax2.set_title(f'Customer Volume by {category_name} (Top 15)', fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plot_path = self.output_dir / f'{category_name.lower()}_churn_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"{category_name} analysis plot saved to: {plot_path}")

    def analyze_churn_over_time(self):
        """Analyze churn patterns over time"""
        print("\n" + "="*60)
        print("CHURN OVER TIME ANALYSIS")
        print("="*60)
        
        if self.merged_df is None:
            raise ValueError("Must create datasets first")
        
        # Convert report_date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.merged_df['report_date']):
            self.merged_df['report_date'] = pd.to_datetime(self.merged_df['report_date'])
        
        # Monthly churn analysis
        monthly_stats = (self.merged_df
                        .groupby([pd.Grouper(key='report_date', freq='M'), 'is_churned'])
                        .size()
                        .unstack(fill_value=0))
        
        if True in monthly_stats.columns and False in monthly_stats.columns:
            monthly_stats['total'] = monthly_stats[True] + monthly_stats[False]
            monthly_stats['churn_rate'] = monthly_stats[True] / monthly_stats['total']
            
            # Create time series visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle('Churn Analysis Over Time', fontsize=16, fontweight='bold')
            
            # Plot 1: Monthly customer counts
            monthly_stats.index = monthly_stats.index.strftime('%Y-%m')
            ax1.plot(monthly_stats.index, monthly_stats[False], marker='o', 
                    label='Active', linewidth=2, color='#27ae60')
            ax1.plot(monthly_stats.index, monthly_stats[True], marker='s', 
                    label='Churned', linewidth=2, color='#e74c3c')
            ax1.set_ylabel('Number of Organizations')
            ax1.set_title('Monthly Organization Counts')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Monthly churn rate
            ax2.plot(monthly_stats.index, monthly_stats['churn_rate'] * 100, 
                    marker='o', linewidth=2, color='#f39c12')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Churn Rate (%)')
            ax2.set_title('Monthly Churn Rate')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Save plot
            plot_path = self.output_dir / 'churn_over_time.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Time series analysis plot saved to: {plot_path}")
            
            # Print summary statistics
            print(f"\nTime Series Summary:")
            print(f"Average monthly churn rate: {monthly_stats['churn_rate'].mean():.1%}")
            print(f"Highest churn rate: {monthly_stats['churn_rate'].max():.1%} in {monthly_stats['churn_rate'].idxmax()}")
            print(f"Lowest churn rate: {monthly_stats['churn_rate'].min():.1%} in {monthly_stats['churn_rate'].idxmin()}")

    def save_datasets(self):
        """Save both datasets"""
        if self.merged_df is not None:
            merged_path = self.output_dir / 'merged_df.csv'
            self.merged_df.to_csv(merged_path, index=False)
            print(f"Merged dataset saved to: {merged_path}")
        
        if self.org_features is not None:
            org_path = self.output_dir / 'org_features.csv'
            self.org_features.to_csv(org_path, index=False)
            print(f"Organization features saved to: {org_path}")

    def run_complete_analysis(self):
        """Run the complete simplified analysis pipeline"""
        print("STARTING SIMPLIFIED CHURN ANALYSIS")
        print("="*60)
        
        try:
            # 1. Create datasets
            merged_df, org_features = self.create_datasets()
            
            # 2. Analyze basic statistics
            self.analyze_basic_statistics()
            
            # 3. Analyze industry and country
            self.analyze_industry_country()
            
            # 4. Analyze churn over time
            self.analyze_churn_over_time()
            
            # 5. Save datasets
            self.save_datasets()
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print("✓ Two datasets created: merged_df and org_features")
            print("✓ Basic statistics analyzed")
            print("✓ Industry and country analysis completed")
            print("✓ Churn over time analysis completed")
            print("✓ All visualizations generated")
            print(f"✓ Results saved to: {self.output_dir}")
            
            return {
                'merged_df': merged_df,
                'org_features': org_features
            }
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            raise


# Example usage:
"""
# Load your datasets
datasets = {
    'commercial': pd.read_csv('commercial_data.csv'),
    'product': pd.read_csv('product_data.csv'),
    # Optional: 'churn': pd.read_csv('churn_data.csv')
}

# Initialize analyzer
analyzer = InitialDataAnalysis(
    datasets=datasets,
    output_dir="churn_analysis_results",
    churn_status_column="churn_status"
)

# Run complete analysis
results = analyzer.run_complete_analysis()

# Access the two main datasets
merged_df = results['merged_df']        # Time-series data
org_features = results['org_features']  # Organization-level features
"""