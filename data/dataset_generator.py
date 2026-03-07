"""
Dataset generator for Electricity Price Forecasting (implementing ENTSOE API)
We use the Belgian market as the target for our analysis
Date Range: 2023-02-01 to 2026-02-01
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class ENTSOEDatasetGenerator:
    # Generates electricity price forecasting datasets from ENTSOE API

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the dataset generator
        
        Args:
            api_token: ENTSOE API token. If None, loads from .env file.
        """
        if api_token is None:
            load_dotenv()
            api_token = os.getenv("securityTokenENTSOE")
        
        if not api_token:
            raise ValueError(
                "ENTSO-E API token not found "
                "Please set 'securityTokenENTSOE' in your .env file or pass it directly"
            )
        
        self.client = EntsoePandasClient(api_key=api_token)
        self.country_price = 'BE'
        self.countries = {
            'BE': 'Belgium',
            'FR': 'France',
            'NL': 'Netherlands',
            'DE': 'Germany'
        }
    
    @staticmethod
    def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure DataFrame index is in UTC timezone
        if df.empty:
            return df
            
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
                return df
            except Exception:
                return df

        if df.index.tz is None:
            return df.tz_localize('UTC')
        return df.tz_convert('UTC')
    
    @staticmethod
    def safe_resample(df: pd.DataFrame, freq: str = 'h') -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        return df.resample(freq).mean()
    
    @staticmethod
    def safe_sum(df: pd.DataFrame, freq: str = 'h') -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        if isinstance(df, pd.Series):
            return df.resample(freq).sum()
        return df.resample(freq).sum().sum(axis=1) if len(df.columns) > 1 else df.resample(freq).sum()
    
    def fetch_data_chunk(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        verbose: bool = True
    ) -> pd.DataFrame:
        # Fetch all data for a given time period

        api_end = end + pd.Timedelta(hours=1)
        
        if verbose:
            print(f"Fetching data from {start.date()} to {end.date()}")
        
        # Get prices
        try:
            prices = self.client.query_day_ahead_prices(
                self.country_price, start=start, end=api_end
            )
            prices_df = prices.to_frame(name="Prices")
            prices_df = self.ensure_utc(prices_df)
            merged = prices_df.copy()
        except Exception as e:
            if verbose:
                print(f"  Error fetching prices: {e}")
            return pd.DataFrame()
        
        # Original exogenous variables (from Jesus Lago)
        # France generation forecast
        try:
            gen_fr_forecast = self.client.query_generation_forecast(
                'FR', start=start, end=api_end
            )
            if isinstance(gen_fr_forecast, pd.Series):
                gen_fr_fc_df = gen_fr_forecast.to_frame(name="FR_Generation_Forecast")
            else:
                gen_fr_fc_df = pd.DataFrame(
                    gen_fr_forecast.sum(axis=1),
                    columns=["FR_Generation_Forecast"]
                )
            gen_fr_fc_df = self.ensure_utc(gen_fr_fc_df)
            gen_fr_fc_hourly = self.safe_resample(gen_fr_fc_df)
            merged = merged.join(gen_fr_fc_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch FR generation forecast: {e}")
        
        # France load forecast
        try:
            load_fr_forecast = self.client.query_load_forecast(
                'FR', start=start, end=api_end
            )
            if isinstance(load_fr_forecast, pd.Series):
                load_fr_fc_df = load_fr_forecast.to_frame(name="FR_Load_Forecast")
            else:
                load_fr_fc_df = pd.DataFrame(
                    load_fr_forecast.sum(axis=1),
                    columns=["FR_Load_Forecast"]
                )
            load_fr_fc_df = self.ensure_utc(load_fr_fc_df)
            load_fr_fc_hourly = self.safe_resample(load_fr_fc_df)
            merged = merged.join(load_fr_fc_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch FR load forecast: {e}")
        
        # other potentially useful variables
        # Belgium load forecast
        try:
            load_be_forecast = self.client.query_load_forecast(
                'BE', start=start, end=api_end
            )
            if isinstance(load_be_forecast, pd.Series):
                load_be_fc_df = load_be_forecast.to_frame(name="BE_Load_Forecast")
            else:
                load_be_fc_df = pd.DataFrame(
                    load_be_forecast.sum(axis=1),
                    columns=["BE_Load_Forecast"]
                )
            load_be_fc_df = self.ensure_utc(load_be_fc_df)
            load_be_fc_hourly = self.safe_resample(load_be_fc_df)
            merged = merged.join(load_be_fc_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch BE load forecast: {e}")
        
        # Belgium actual load
        try:
            load_be_actual = self.client.query_load('BE', start=start, end=api_end)
            if isinstance(load_be_actual, pd.Series):
                load_be_act_df = load_be_actual.to_frame(name="BE_Load_Actual")
            else:
                load_be_act_df = pd.DataFrame(
                    load_be_actual.sum(axis=1),
                    columns=["BE_Load_Actual"]
                )
            load_be_act_df = self.ensure_utc(load_be_act_df)
            load_be_act_hourly = self.safe_resample(load_be_act_df)
            merged = merged.join(load_be_act_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch BE actual load: {e}")
        
        # Belgium generation forecast
        try:
            gen_be_forecast = self.client.query_generation_forecast(
                'BE', start=start, end=api_end
            )
            if isinstance(gen_be_forecast, pd.Series):
                gen_be_fc_df = gen_be_forecast.to_frame(name="BE_Generation_Forecast")
            else:
                gen_be_fc_df = pd.DataFrame(
                    gen_be_forecast.sum(axis=1),
                    columns=["BE_Generation_Forecast"]
                )
            gen_be_fc_df = self.ensure_utc(gen_be_fc_df)
            gen_be_fc_hourly = self.safe_resample(gen_be_fc_df)
            merged = merged.join(gen_be_fc_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch BE generation forecast: {e}")
        
        # Belgium actual generation
        try:
            gen_be_actual = self.client.query_generation('BE', start=start, end=api_end)
            if isinstance(gen_be_actual, pd.Series):
                gen_be_act_df = gen_be_actual.to_frame(name="BE_Generation_Actual")
            else:
                gen_be_act_df = pd.DataFrame(
                    gen_be_actual.sum(axis=1),
                    columns=["BE_Generation_Actual"]
                )
            gen_be_act_df = self.ensure_utc(gen_be_act_df)
            gen_be_act_hourly = self.safe_resample(gen_be_act_df)
            merged = merged.join(gen_be_act_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch BE actual generation: {e}")
        
        # Belgium wind generation forecast
        try:
            gen_wind_be = self.client.query_wind_and_solar_forecast(
                'BE', start=start, end=api_end, psr_type='B16'
            )
            if isinstance(gen_wind_be, pd.Series):
                wind_be_df = gen_wind_be.to_frame(name="BE_Wind_Forecast")
            else:
                wind_be_df = pd.DataFrame(
                    gen_wind_be.sum(axis=1),
                    columns=["BE_Wind_Forecast"]
                ) if not gen_wind_be.empty else pd.DataFrame()
            if not wind_be_df.empty:
                wind_be_df = self.ensure_utc(wind_be_df)
                wind_be_hourly = self.safe_resample(wind_be_df)
                merged = merged.join(wind_be_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch BE wind forecast: {e}")
        
        # Belgium solar generation forecast
        try:
            gen_solar_be = self.client.query_wind_and_solar_forecast(
                'BE', start=start, end=api_end, psr_type='B18'
            )
            if isinstance(gen_solar_be, pd.Series):
                solar_be_df = gen_solar_be.to_frame(name="BE_Solar_Forecast")
            else:
                solar_be_df = pd.DataFrame(
                    gen_solar_be.sum(axis=1),
                    columns=["BE_Solar_Forecast"]
                ) if not gen_solar_be.empty else pd.DataFrame()
            if not solar_be_df.empty:
                solar_be_df = self.ensure_utc(solar_be_df)
                solar_be_hourly = self.safe_resample(solar_be_df)
                merged = merged.join(solar_be_hourly, how="left")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch BE solar forecast: {e}")
        
        # Neighboring country prices
        for country_code in ['FR', 'NL', 'DE']:
            try:
                prices_neighbor = self.client.query_day_ahead_prices(
                    country_code, start=start, end=api_end
                )
                prices_neighbor_df = prices_neighbor.to_frame(
                    name=f"{country_code}_Prices"
                )
                prices_neighbor_df = self.ensure_utc(prices_neighbor_df)
                merged = merged.join(prices_neighbor_df, how="left")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not fetch {country_code} prices: {e}")
        
        # Cross-border flows
        for neighbor in ['FR', 'NL', 'DE']:
            try:
                flow = self.client.query_crossborder_flows(
                    self.country_price, neighbor, start=start, end=api_end
                )
                flow_df = flow.to_frame(name=f"Flow_BE_{neighbor}")
                flow_df = self.ensure_utc(flow_df)
                flow_hourly = self.safe_resample(flow_df)
                merged = merged.join(flow_hourly, how="left")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not fetch BE-{neighbor} flow: {e}")
        
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        
        return merged
    
    def fetch_full_dataset(
        self,
        start_date: str = "2023-02-01",
        end_date: str = "2026-02-01",
        verbose: bool = True
    ) -> pd.DataFrame:
        # Fetch the complete dataset by querying year by year (otherwise ENTSOE API doesn't work)

        start = pd.Timestamp(f"{start_date}T00:00Z")
        end = pd.Timestamp(f"{end_date}T00:00Z")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Fetching dataset from {start_date} to {end_date}")
            print(f"{'='*80}\n")
        
        all_data = []
        
        # Query year by year to avoid API limits
        for year in range(start.year, end.year + 1):
            query_start = pd.Timestamp(f"{year}-01-01T00:00Z")
            query_end = pd.Timestamp(f"{year}-12-31T23:59Z")
            
            if query_start < start:
                query_start = start
            if query_end > end:
                query_end = end
            
            if query_start >= query_end:
                continue
            
            try:
                chunk = self.fetch_data_chunk(query_start, query_end, verbose=verbose)
                if not chunk.empty:
                    all_data.append(chunk)
                    if verbose:
                        print(f"  ✓ Retrieved {len(chunk)} hourly data points for {year}\n")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error fetching data for {year}: {e}\n")
            
            time.sleep(1)  # Rate limiting
        
        if not all_data:
            raise ValueError("No data was retrieved. Please check your date range and API token.")
        
        # Combine all chunks
        final_df = pd.concat(all_data)
        final_df = final_df.sort_index()
        final_df = final_df[~final_df.index.duplicated(keep="first")]
        
        # Resample to hourly (ENTSOE switched from hourly to 15-min on ~30/09/2025)
        # Keeps data consistent: aggregate 15-min points to hourly via mean
        final_df = self.safe_resample(final_df, freq='1h')
        final_df = final_df.dropna(how='all')
        
        # Clip to exact date range
        final_df = final_df.loc[(final_df.index >= start) & (final_df.index < end)]
        
        # Remove timezone info
        final_df.index = final_df.index.tz_convert(None)
        
        if verbose:
            print(f"{'='*80}")
            print(f"Dataset retrieval complete!")
            print(f"  Total data points: {len(final_df)}")
            print(f"  Date range: {final_df.index.min()} to {final_df.index.max()}")
            print(f"  Variables: {len(final_df.columns)}")
            print(f"{'='*80}\n")
        
        return final_df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add time-based features to the dataset

        df = df.copy()
        # Hour of day
        df['Hour'] = df.index.hour
        # Day of week (0=monday-6=sunday)
        df['DayOfWeek'] = df.index.dayofweek
        # Month (1-12)
        df['Month'] = df.index.month
        # Is weekend (boolean)
        df['IsWeekend'] = (df.index.dayofweek >= 5).astype(int)
        # Day of year (1-365)
        df['DayOfYear'] = df.index.dayofyear
        # Week of year
        df['WeekOfYear'] = df.index.isocalendar().week
        # Cyclical encoding for hour (sine and cosine)
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        # Cyclical encoding for day of week
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        # Cyclical encoding for month
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add derived features (like imbalances, net positions, etc.)

        df = df.copy()
        
        # Load imbalance (actual - forecast)
        if 'BE_Load_Actual' in df.columns and 'BE_Load_Forecast' in df.columns:
            df['BE_Load_Imbalance'] = (
                df['BE_Load_Actual'] - df['BE_Load_Forecast']
            )
        
        # Generation imbalance
        if ('BE_Generation_Actual' in df.columns and 
            'BE_Generation_Forecast' in df.columns):
            df['BE_Generation_Imbalance'] = (
                df['BE_Generation_Actual'] - df['BE_Generation_Forecast']
            )
        
        # Net position (generation - load)
        if ('BE_Generation_Forecast' in df.columns and 
            'BE_Load_Forecast' in df.columns):
            df['BE_Net_Position'] = (
                df['BE_Generation_Forecast'] - df['BE_Load_Forecast']
            )
        
        # Price spreads (differences with neighbors)
        for country in ['FR', 'NL', 'DE']:
            price_col = f"{country}_Prices"
            if price_col in df.columns:
                df[f"Price_Spread_{country}"] = df['Prices'] - df[price_col]
        
        return df
    
    def validate_variables(
        self,
        df: pd.DataFrame,
        train_split_date: str = "2023-02-01",
        test_start_date: Optional[str] = None,
        top_n: int = 15
    ) -> pd.DataFrame:
        """
        Validate exogenous variables using correlation and mutual information
        
        Args:
            df: full dataset
            train_split_date: Date to split training data (for validation)
            test_start_date: Start of test set (if None uses the last 6 months)
            top_n: Number of top variables to return
            
        Returns:
            DataFrame with validation metrics for each variable
        """
        print(f"\n{'='*80}")
        print("Validating Exogenous Variables")
        print(f"{'='*80}\n")
        
        # Split data for validation (use training portion only)
        df_indexed = df.set_index(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df
        
        if test_start_date is None:
            # Use last 6 months as test (like config)
            test_start = df_indexed.index.max() - pd.DateOffset(months=6)
        else:
            test_start = pd.to_datetime(test_start_date)
        
        train_df = df_indexed.loc[df_indexed.index < test_start].copy()
        
        print(f"Using training period: {train_df.index.min()} to {train_df.index.max()}")
        print(f"Training samples: {len(train_df)}\n")
        
        # Get numeric columns (exclude prices)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Prices' in numeric_cols:
            numeric_cols.remove('Prices')
        
        # Remove columns with too many missing values
        valid_cols = []
        for col in numeric_cols:
            missing_pct = train_df[col].isna().sum() / len(train_df)
            if missing_pct < 0.5:
                valid_cols.append(col)
            else:
                print(f"  Skipping {col}: {missing_pct*100:.1f}% missing")
        
        print(f"\nAnalyzing {len(valid_cols)} variables\n")


        prices = train_df['Prices'].values
        results = []
        
        for col in valid_cols:
            # Align data and remove NaN pairs
            aligned = train_df[['Prices', col]].dropna()
            
            if len(aligned) < 100:  # not enough data
                continue
            
            prices_clean = aligned['Prices'].values
            var_clean = aligned[col].values
            
            # Calculate correlation
            corr = np.corrcoef(prices_clean, var_clean)[0, 1]
            
            # Calculate mutual information (with scaling)
            scaler = StandardScaler()
            var_scaled = scaler.fit_transform(var_clean.reshape(-1, 1)).ravel()
            prices_scaled = scaler.fit_transform(prices_clean.reshape(-1, 1)).ravel()
            
            # Discretize into 10 bins for mutual information
            var_binned = pd.cut(var_scaled, bins=10, labels=False, duplicates='drop')
            prices_binned = pd.cut(prices_scaled, bins=10, labels=False, duplicates='drop')
            
            # Remove any NaN
            valid_mask = ~(np.isnan(var_binned) | np.isnan(prices_binned))
            if valid_mask.sum() < 100:
                mi = 0.0
            else:
                mi = mutual_info_regression(
                    var_binned[valid_mask].reshape(-1, 1),
                    prices_binned[valid_mask],
                    random_state=42
                )[0]
            
            # Calculate missing percentage
            missing_pct = train_df[col].isna().sum() / len(train_df)
            
            results.append({
                'Variable': col,
                'Correlation': corr,
                'Abs_Correlation': abs(corr),
                'Mutual_Info': mi,
                'Missing_Pct': missing_pct,
                'N_Samples': len(aligned)
            })
        
        results_df = pd.DataFrame(results)
        
        # Sort by absolute correlation and mutual information
        results_df = results_df.sort_values(
            ['Abs_Correlation', 'Mutual_Info'],
            ascending=[False, False]
        )
        
        print(f"{'Variable':<40} {'Correlation':>12} {'Mutual Info':>12} {'Missing %':>10}")
        print("-" * 80)
        
        for _, row in results_df.head(top_n).iterrows():
            print(
                f"{row['Variable']:<40} "
                f"{row['Correlation']:>12.4f} "
                f"{row['Mutual_Info']:>12.4f} "
                f"{row['Missing_Pct']*100:>9.1f}%"
            )
        
        print(f"\n{'='*80}")
        print(f"Top {min(top_n, len(results_df))} variables selected")
        print(f"{'='*80}\n")
        
        return results_df
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: str,
        selected_variables: Optional[List[str]] = None
    ) -> None:
        #Save dataset to CSV file
        df_save = df.copy()
        
        # Select variables if there are specified
        if selected_variables is not None:
            if 'Prices' not in selected_variables:
                selected_variables = ['Prices'] + selected_variables
            
            available_vars = [v for v in selected_variables if v in df_save.columns]
            df_save = df_save[available_vars]
        
        df_save = df_save.reset_index()
        if 'index' in df_save.columns:
            df_save = df_save.rename(columns={'index': 'Date'})
        
        if 'Date' not in df_save.columns and isinstance(df.index, pd.DatetimeIndex):
            df_save.insert(0, 'Date', df.index)
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df_save.to_csv(output_path, index=False)
        
        print(f"\n{'='*50}")
        print(f"Dataset saved successfully!")
        print(f"  Path: {output_path}")
        print(f"  Shape: {df_save.shape}")
        print(f"  Variables: {len(df_save.columns) - 1} (excluding Date)")
        print(f"  Date range: {df_save['Date'].min()} to {df_save['Date'].max()}")
        print(f"{'='*50}\n")


def main():
    # Initialize generator
    generator = ENTSOEDatasetGenerator()
    
    # Fetch full dataset
    df = generator.fetch_full_dataset(
        start_date="2023-02-01",
        end_date="2026-02-01",
        verbose=True
    )
    
    # Add time features
    print("Adding time-based features")
    df = generator.add_time_features(df)
    
    # Add derived features
    print("Adding derived features")
    df = generator.add_derived_features(df)
    
    # Validate variables
    validation_results = generator.validate_variables(
        df,
        train_split_date="2023-02-01",
        top_n=20
    )

    time_features = [
        'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'DayOfYear', 'WeekOfYear',
        'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
        'Month_sin', 'Month_cos'
    ]
    
    # Get top exogenous variables (excluding time features)
    top_exog = validation_results[
        ~validation_results['Variable'].isin(time_features + ['Prices'])
    ].head(15)['Variable'].tolist()
    
    # use prices + time features + top exogenous variables
    selected_vars = ['Prices'] + time_features + top_exog
    
    print(f"\nSelected {len(selected_vars)} variables:")
    print(f"  - Target: Prices")
    print(f"  - Time features: {len(time_features)}")
    print(f"  - Exogenous variables: {len(top_exog)}")
    print(f"\nExogenous variables:")
    for var in top_exog:
        print(f"  - {var}")
    
    # Save full dataset
    output_dir = Path(__file__).parent.parent / "data" / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    generator.save_dataset(
        df,
        output_dir / "BE_ENTSOE_FULL.csv",
        selected_variables=None  # Save all variables
    )
    
    # Save selected variables dataset
    generator.save_dataset(
        df,
        output_dir / "BE_ENTSOE.csv",
        selected_variables=selected_vars
    )
    
    # Save validation results
    validation_results.to_csv(
        output_dir / "variable_validation.csv",
        index=False
    )
    
    print("\nDataset generation complete!")
    print(f"\nFiles created:")
    print(f"  1. {output_dir / 'BE_ENTSOE_FULL.csv'} - All variables")
    print(f"  2. {output_dir / 'BE_ENTSOE.csv'} - Selected variables")
    print(f"  3. {output_dir / 'variable_validation.csv'} - Validation metrics")


if __name__ == "__main__":
    main()
