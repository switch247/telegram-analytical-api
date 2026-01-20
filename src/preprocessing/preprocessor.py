"""

This class cleans and preprocesses the scraped reviews data.
- Handles missing values
- Normalizes dates
- Cleans text data
"""

import sys
import os
import re
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.config.settings import DATA_PATHS


@dataclass
class PreprocessSchema:
    """Column mapping to keep the pipeline dataset-agnostic."""

    text_col: str = 'review_text'
    rating_col: str = 'rating'
    date_col: str = 'review_date'
    id_col: str = 'review_id'
    bank_code_col: str = 'bank_code'
    bank_name_col: str = 'bank_name'
    user_col: str = 'user_name'
    thumbs_up_col: str = 'thumbs_up'
    reply_col: str = 'reply_content'
    source_col: str = 'source'
    year_col: str = 'review_year'
    month_col: str = 'review_month'
    text_length_col: str = 'text_length'
    sort_cols: tuple = (('bank_code', True), ('review_date', False))

    def output_columns(self):
        return [
            col
            for col in [
                self.id_col,
                self.text_col,
                self.rating_col,
                self.date_col,
                self.year_col,
                self.month_col,
                self.bank_code_col,
                self.bank_name_col,
                self.user_col,
                self.thumbs_up_col,
                self.text_length_col,
                self.source_col,
            ]
            if col
        ]


class DatasetPreprocessor:
    """Dataset-agnostic preprocessor driven by a configurable column schema."""

    def __init__(self, input_path=None, output_path=None, schema=None, critical_cols=None, verbose=True):
        """Initialize preprocessor with optional paths, schema, and verbosity."""
        self.input_path = input_path or DATA_PATHS['raw_reviews']
        self.output_path = output_path or DATA_PATHS['processed_reviews']
        self.schema = self._init_schema(schema)
        self.schema.sort_cols = self._normalize_sort_cols(self.schema.sort_cols)
        self.critical_cols = critical_cols or [
            col for col in (self.schema.text_col, self.schema.rating_col, self.schema.bank_name_col) if col
        ]
        self.verbose = verbose
        self.df = None
        self.stats = {}

    def _log(self, message):
        """Print helper that can be silenced for reuse/testing"""
        if self.verbose:
            print(message)

    def _init_schema(self, schema):
        """Normalize schema input to PreprocessSchema instance."""
        if schema is None:
            return PreprocessSchema()
        if isinstance(schema, PreprocessSchema):
            return schema
        if isinstance(schema, dict):
            return PreprocessSchema(**schema)
        raise ValueError("schema must be None, dict, or PreprocessSchema instance")

    def _normalize_sort_cols(self, sort_cols):
        """Ensure sort columns are a list of (col, ascending) tuples."""
        normalized = []
        if not sort_cols:
            return normalized
        for item in sort_cols:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append((item[0], bool(item[1])))
            else:
                normalized.append((item, True))
        return tuple(normalized)

    def reset_stats(self):
        """Clear cached stats before a fresh run"""
        self.stats = {}

    def load_data(self):
        """Load raw reviews data"""
        # Print a message indicating that data loading has started
        self._log("Loading raw data...")
        try:
            # Read the CSV file at self.input_path into a pandas DataFrame
            self.df = pd.read_csv(self.input_path)
            # Print the number of records loaded
            self._log(f"Loaded {len(self.df)} reviews")
            # Record the initial number of records in our stats dictionary
            self.stats['original_count'] = len(self.df)
            # Return True to indicate success
            return True
        except FileNotFoundError:
            # Handle the specific error where the file does not exist
            self._log(f"ERROR: File not found: {self.input_path}")
            # Return False to indicate failure
            return False
        except Exception as e:
            # Handle any other general errors that occur during loading
            self._log(f"ERROR: Failed to load data: {str(e)}")
            # Return False to indicate failure
            return False

    def check_missing_data(self):
        """Check for missing data"""
        # Print a header for this step [1/6]
        self._log("\n[1/6] Checking for missing data...")

        # Calculate the count of missing (null) values for each column
        missing = self.df.isnull().sum()
        # Calculate the percentage of missing values for each column
        missing_pct = (missing / len(self.df)) * 100

        # Print the section header
        self._log("\nMissing values:")
        # Loop through each column name in the index of the 'missing' series
        for col in missing.index:
            # If the column has at least one missing value
            if missing[col] > 0:
                # Print the column name, count of missing values, and percentage
                self._log(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

        # Store the dictionary of missing counts in our stats for reporting later
        self.stats['missing_before'] = missing.to_dict()

        # Define a list of columns that are absolutely required for our analysis
        critical_cols = self.critical_cols
        # Calculate missing values just for these critical columns
        missing_critical = self.df[critical_cols].isnull().sum()

        # If there are any missing values in critical columns
        if missing_critical.sum() > 0:
            # Print a warning message
            self._log("\nWARNING: Missing values in critical columns:")
            # Print the counts of missing values for the critical columns that have them
            self._log(missing_critical[missing_critical > 0])

    def handle_missing_values(self):
        """Handle missing values"""
        # Print a header for this step [2/6]
        self._log("\n[2/6] Handling missing values...")

        # Define the critical columns again
        critical_cols = [col for col in self.critical_cols if col in self.df.columns]
        # Store the count before dropping rows
        before_count = len(self.df)
        # Drop any rows that have missing values (NaN) in the critical columns
        if critical_cols:
            self.df = self.df.dropna(subset=critical_cols)
        # Calculate how many rows were removed
        removed = before_count - len(self.df)

        # If any rows were removed, print a message
        if removed > 0:
            self._log(f"Removed {removed} rows with missing critical values")

        # For the optional user-related columns, fill with sensible defaults when present
        if self.schema.user_col in self.df.columns:
            self.df[self.schema.user_col] = self.df[self.schema.user_col].fillna('Anonymous')
        if self.schema.thumbs_up_col in self.df.columns:
            self.df[self.schema.thumbs_up_col] = self.df[self.schema.thumbs_up_col].fillna(0)
        if self.schema.reply_col in self.df.columns:
            self.df[self.schema.reply_col] = self.df[self.schema.reply_col].fillna('')

        # Record the number of rows removed due to missing critical data
        self.stats['rows_removed_missing'] = removed
        # Record the new total count in stats
        self.stats['count_after_missing'] = len(self.df)

    def normalize_dates(self):
        """Normalize date formats to YYYY-MM-DD"""
        # Print a header for this step [3/6]
        self._log("\n[3/6] Normalizing dates...")

        if self.schema.date_col not in self.df.columns:
            self._log("WARNING: Date column not found; skipping date normalization")
            return

        try:
            # Convert the 'review_date' column to pandas datetime objects
            # This handles various string formats automatically
            self.df[self.schema.date_col] = pd.to_datetime(self.df[self.schema.date_col])

            # Convert the datetime objects to just date objects (YYYY-MM-DD), removing time info
            self.df[self.schema.date_col] = self.df[self.schema.date_col].dt.date

            # Extract the year from the date and create a new 'review_year' column
            self.df[self.schema.year_col] = pd.to_datetime(self.df[self.schema.date_col]).dt.year
            # Extract the month from the date and create a new 'review_month' column
            self.df[self.schema.month_col] = pd.to_datetime(self.df[self.schema.date_col]).dt.month

            # Print the range of dates found in the data (minimum and maximum)
            self._log(f"Date range: {self.df[self.schema.date_col].min()} to {self.df[self.schema.date_col].max()}")

        except Exception as e:
            # Handle errors if date conversion fails
            self._log(f"WARNING: Error normalizing dates: {str(e)}")

    def clean_text(self):
        """Clean review text"""
        # Print a header for this step [4/6]
        self._log("\n[4/6] Cleaning text...")

        if self.schema.text_col not in self.df.columns:
            self._log("WARNING: Text column not found; skipping text cleaning")
            return

        def clean_review_text(text):
            """Inner function to clean individual review text strings"""
            # If the text is NaN (missing) or empty string, return empty string
            if pd.isna(text) or text == '':
                return ''

            # Convert the input to a string type (safety check)
            text = str(text)

            # Use regex to replace multiple whitespace characters (spaces, tabs, newlines) with a single space
            text = re.sub(r'\s+', ' ', text)

            # Remove leading and trailing whitespace from the string
            text = text.strip()

            # Return the cleaned text
            return text

        # Apply the 'clean_review_text' function to every element in the 'review_text' column
        self.df[self.schema.text_col] = self.df[self.schema.text_col].apply(clean_review_text)

        # Store the count before removing empty reviews
        before_count = len(self.df)
        # Keep only rows where the length of 'review_text' is greater than 0
        self.df = self.df[self.df[self.schema.text_col].str.len() > 0]
        # Calculate how many empty reviews were removed
        removed = before_count - len(self.df)

        # If rows were removed, print a message
        if removed > 0:
            self._log(f"Removed {removed} reviews with empty text")

        # Create a new column 'text_length' containing the character count of the review text
        self.df[self.schema.text_length_col] = self.df[self.schema.text_col].str.len()

        # Record statistics about text cleaning
        self.stats['empty_reviews_removed'] = removed
        self.stats['count_after_cleaning'] = len(self.df)

    def validate_ratings(self):
        """Validate rating values (should be 1-5)"""
        # Print a header for this step [5/6]
        self._log("\n[5/6] Validating ratings...")

        if self.schema.rating_col not in self.df.columns:
            self._log("WARNING: Rating column not found; skipping rating validation")
            return

        # Find rows where 'rating' is less than 1 OR greater than 5
        invalid = self.df[(self.df[self.schema.rating_col] < 1) | (self.df[self.schema.rating_col] > 5)]

        # If there are any invalid ratings
        if len(invalid) > 0:
            # Print a warning with the count of invalid ratings
            self._log(f"WARNING: Found {len(invalid)} reviews with invalid ratings")
            # Filter the DataFrame to keep only rows where rating is between 1 and 5 (inclusive)
            self.df = self.df[(self.df[self.schema.rating_col] >= 1) & (self.df[self.schema.rating_col] <= 5)]
        else:
            # If all ratings are valid, print a confirmation
            self._log("All ratings are valid (1-5)")

        # Record the number of invalid ratings removed
        self.stats['invalid_ratings_removed'] = len(invalid)

    def prepare_final_output(self):
        """Prepare final output format"""
        # Print a header for this step [6/6]
        self._log("\n[6/6] Preparing final output...")

        output_columns = [col for col in self.schema.output_columns() if col in self.df.columns]
        if output_columns:
            self.df = self.df[output_columns]

        sort_cols = [(col, asc) for col, asc in self.schema.sort_cols if col in self.df.columns]
        if sort_cols:
            cols, ascending = zip(*sort_cols)
            self.df = self.df.sort_values(list(cols), ascending=list(ascending))

        # Reset the index of the DataFrame so it starts from 0 to N-1 cleanly
        # drop=True prevents the old index from being added as a new column
        self.df = self.df.reset_index(drop=True)

        # Print the final count of reviews
        self._log(f"Final dataset: {len(self.df)} reviews")
        self.stats['final_count'] = len(self.df)

    def process_dataframe(self, df, *, output_path=None, save=False, report=False):
        """Run preprocessing on a provided DataFrame and optionally persist results."""
        self.reset_stats()
        self.df = df.copy()
        self.stats['original_count'] = len(self.df)

        self.check_missing_data()
        self.handle_missing_values()
        self.normalize_dates()
        self.clean_text()
        self.validate_ratings()
        self.prepare_final_output()

        if save:
            if output_path:
                self.output_path = output_path
            if not self.save_data():
                return None

        if report:
            self.generate_report()

        return self.df

    def save_data(self):
        """Save processed data"""
        # Print a message indicating saving has started
        self._log("\nSaving processed data...")

        try:
            # Create the directory for the output file if it doesn't already exist
            # os.path.dirname gets the folder part of the file path
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

            # Write the DataFrame to a CSV file at self.output_path
            # index=False prevents writing the row numbers (0, 1, 2...) to the file
            self.df.to_csv(self.output_path, index=False)
            # Print a confirmation message with the path
            self._log(f"Data saved to: {self.output_path}")

            # Record the final count in stats
            self.stats['final_count'] = len(self.df)
            # Return True to indicate success
            return True

        except Exception as e:
            # Handle any errors during saving
            self._log(f"ERROR: Failed to save data: {str(e)}")
            # Return False to indicate failure
            return False

    def generate_report(self):
        """Generate preprocessing report"""
        # Print a separator line
        self._log("\n" + "=" * 60)
        # Print the report title
        self._log("PREPROCESSING REPORT")
        # Print a separator line
        self._log("=" * 60)

        # Print various statistics gathered during the process using .get() to avoid errors if key is missing
        self._log(f"\nOriginal records: {self.stats.get('original_count', 0)}")
        self._log(f"Records with missing critical data: {self.stats.get('rows_removed_missing', 0)}")
        self._log(f"Empty reviews removed: {self.stats.get('empty_reviews_removed', 0)}")
        self._log(f"Invalid ratings removed: {self.stats.get('invalid_ratings_removed', 0)}")
        self._log(f"Final records: {self.stats.get('final_count', 0)}")

        # Calculate data quality percentage metrics
        if self.stats.get('original_count', 0) > 0:
            # Retention rate = (Final / Original) * 100
            # We use .get(..., 1) for denominator to avoid division by zero if original_count is missing
            retention_rate = (self.stats.get('final_count', 0) / self.stats.get('original_count', 1)) * 100
            # Error rate is the inverse of retention rate
            error_rate = 100 - retention_rate
            self._log(f"\nData retention rate: {retention_rate:.2f}%")
            self._log(f"Data error rate: {error_rate:.2f}%")

            # Assess quality based on error rate thresholds
            if error_rate < 5:
                self._log("✓ Data quality: EXCELLENT (<5% errors)")
            elif error_rate < 10:
                self._log("✓ Data quality: GOOD (<10% errors)")
            else:
                self._log("⚠ Data quality: NEEDS ATTENTION (>10% errors)")

        # Print statistics about the reviews per bank
        if self.df is not None:
            if self.schema.bank_name_col in self.df.columns:
                self._log("\nRecords per category:")
                bank_counts = self.df[self.schema.bank_name_col].value_counts()
                for bank, count in bank_counts.items():
                    self._log(f"  {bank}: {count}")

            if self.schema.rating_col in self.df.columns:
                self._log("\nRating distribution:")
                rating_counts = self.df[self.schema.rating_col].value_counts().sort_index(ascending=False)
                for rating, count in rating_counts.items():
                    pct = (count / len(self.df)) * 100
                    self._log(f"  {'⭐' * int(rating)}: {count} ({pct:.1f}%)")

            if self.schema.date_col in self.df.columns:
                self._log(
                    f"\nDate range: {self.df[self.schema.date_col].min()} to {self.df[self.schema.date_col].max()}"
                )

            if self.schema.text_length_col in self.df.columns:
                self._log(f"\nText statistics:")
                self._log(f"  Average length: {self.df[self.schema.text_length_col].mean():.0f} characters")
                self._log(f"  Median length: {self.df[self.schema.text_length_col].median():.0f} characters")
                self._log(f"  Min length: {self.df[self.schema.text_length_col].min()}")
                self._log(f"  Max length: {self.df[self.schema.text_length_col].max()}")

    def process(self):
        """Run complete preprocessing pipeline"""
        # Print start header
        self._log("=" * 60)
        self._log("STARTING DATA PREPROCESSING")
        self._log("=" * 60)

        # Attempt to load data. If it fails, return False immediately.
        if not self.load_data():
            return False

        result = self.process_dataframe(self.df, save=True, report=True)
        return result is not None


if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    preprocessor.process()
