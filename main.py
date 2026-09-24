"""
Player Matcher: A modular framework for fuzzy matching player records across different data providers.

This module provides tools to match player records from different sources using TF-IDF vectorization
and bidirectional matching to ensure high-quality matches.
"""

import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import re
import time
import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for a data provider."""
    name: str
    player_name_col: str = 'player_name'
    player_id_col: str = 'player_id'
    date_of_birth_col: str = 'date_of_birth'


class PlayerNameProcessor:
    """Handles player name cleaning and date of birth encoding."""

    MONTH_MAPPING = {
        '01': 'A', '02': 'B', '03': 'C', '04': 'D', '05': 'E',
        '06': 'F', '07': 'G', '08': 'H', '09': 'I', '10': 'J',
        '11': 'K', '12': 'L'
    }

    @staticmethod
    def clean_name(name: str) -> Optional[str]:
        """
        Clean player name by removing punctuation, normalizing spaces, and lowercasing.

        Args:
            name: Raw player name

        Returns:
            Cleaned name or None if input is invalid
        """
        if name and name == name:  # Check for non-null and non-NaN
            return re.sub(r'\W+', ' ', unidecode(name)).strip().replace('  ', ' ').lower()
        return None

    @staticmethod
    def convert_date(date_string: str) -> str:
        """
        Convert date from YYYY-MM-DD format to DDMYY format (e.g., 1990-03-15 -> 15C90).

        Args:
            date_string: Date in YYYY-MM-DD format

        Returns:
            Encoded date string or '00000' if invalid
        """
        try:
            if date_string and date_string == date_string:
                year, month, day = date_string.split('-')
                month_letter = PlayerNameProcessor.MONTH_MAPPING[month]
                year_last_two = year[-2:]
                day = day.zfill(2)
                return f"{day}{month_letter}{year_last_two}"
        except:
            pass
        return '00000'

    @staticmethod
    def create_searchable_name(name: str, date_of_birth: str) -> str:
        """
        Create a searchable name by combining cleaned name with encoded date of birth.

        Args:
            name: Player name
            date_of_birth: Date of birth in YYYY-MM-DD format

        Returns:
            Combined searchable string
        """
        cleaned_name = PlayerNameProcessor.clean_name(name)
        encoded_dob = PlayerNameProcessor.convert_date(date_of_birth)
        return f"{cleaned_name} {encoded_dob}" if cleaned_name else encoded_dob


class TfidfMatcher:
    """Performs TF-IDF based fuzzy matching between two datasets."""

    def __init__(self, ngram_size: int = 3, min_similarity: float = 0.5):
        """
        Initialize the matcher.

        Args:
            ngram_size: Size of n-grams for TF-IDF vectorization
            min_similarity: Minimum similarity threshold for matches
        """
        self.ngram_size = ngram_size
        self.min_similarity = min_similarity
        self.vectorizer = None

    def _ngrams(self, string: str, n: int = None) -> List[str]:
        """Generate n-grams from a string."""
        n = n or self.ngram_size
        string = re.sub(r"[,-./]|\s", r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def fit_vectorizer(self, names: pd.Series) -> None:
        """
        Fit the TF-IDF vectorizer on the provided names.

        Args:
            names: Series of names to fit the vectorizer
        """
        self.vectorizer = TfidfVectorizer(analyzer=self._ngrams)
        self.vectorizer.fit(names)

    def match_most_similar(
        self,
        master_name: pd.Series,
        master_id: pd.Series,
        duplicates_name: pd.Series,
        duplicates_id: pd.Series,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Find the most similar match for each duplicate in the master dataset.

        Args:
            master_name: Series of master names
            master_id: Series of master IDs
            duplicates_name: Series of duplicate names
            duplicates_id: Series of duplicate IDs
            verbose: Whether to print progress messages

        Returns:
            DataFrame with matches including similarity scores
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before matching. Call fit_vectorizer() first.")

        right_matrix = self.vectorizer.transform(master_name)
        left_matrix = self.vectorizer.transform(duplicates_name)

        if verbose:
            print('\tPerforming similarity matching...')
        start_time = time.time()
        matches = sp_matmul_topn(
            left_matrix,
            right_matrix.T,
            top_n=1,
            threshold=self.min_similarity
        )
        if verbose:
            print(f'\tMatching completed in {time.time() - start_time:.2f} seconds.')

        # Extract match indices and similarities
        r, c = matches.nonzero()
        similarities = matches.data

        matches_df = pd.DataFrame({
            'duplicate_idx': r.astype(np.int64),
            'master_idx': c.astype(np.int64),
            'similarity': similarities
        })

        # Keep only the best match for each duplicate
        matches_df = matches_df.loc[
            matches_df.groupby('duplicate_idx')['similarity'].idxmax()
        ]

        # Add IDs and names
        result = matches_df.copy()
        result['duplicate_id'] = duplicates_id.iloc[matches_df['duplicate_idx']].values
        result['master_id'] = master_id.iloc[matches_df['master_idx']].values
        result['duplicate_name'] = duplicates_name.iloc[matches_df['duplicate_idx']].values
        result['master_name'] = master_name.iloc[matches_df['master_idx']].values

        return result


class BidirectionalPlayerMatcher:
    """
    Matches player records between two providers using bidirectional fuzzy matching.

    This ensures that matches are mutual (A matches B AND B matches A) for higher quality.
    """

    def __init__(
        self,
        provider_a: ProviderConfig,
        provider_b: ProviderConfig,
        min_similarity: float = 0.5,
        ngram_size: int = 3
    ):
        """
        Initialize the bidirectional matcher.

        Args:
            provider_a: Configuration for first provider
            provider_b: Configuration for second provider
            min_similarity: Minimum similarity threshold
            ngram_size: Size of n-grams for matching
        """
        if provider_a.name == provider_b.name:
            raise ValueError(
                f"provider_a and provider_b must have different names (both are '{provider_a.name}'): "
                "the names are used as suffixes for the output columns."
            )
        self.provider_a = provider_a
        self.provider_b = provider_b
        self.matcher = TfidfMatcher(ngram_size=ngram_size, min_similarity=min_similarity)
        self.name_processor = PlayerNameProcessor()

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        provider: ProviderConfig
    ) -> pd.DataFrame:
        """
        Prepare a dataset by cleaning names and creating searchable names.

        Args:
            df: Input dataframe
            provider: Provider configuration

        Returns:
            Prepared dataframe with additional columns
        """
        df = df.copy()

        # Create cleaned name column
        df['_cleaned_name'] = df[provider.player_name_col].apply(
            self.name_processor.clean_name
        )

        # Create searchable name (name + encoded DOB)
        df['_searchable_name'] = df.apply(
            lambda row: self.name_processor.create_searchable_name(
                row[provider.player_name_col],
                row[provider.date_of_birth_col]
            ),
            axis=1
        )

        return df

    def find_bidirectional_matches(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Find bidirectional matches between two datasets.

        Args:
            df_a: First dataset (already prepared)
            df_b: Second dataset (already prepared)
            verbose: Whether to print progress messages

        Returns:
            DataFrame with matched pairs
        """
        # Filter out rows with missing data
        df_a_clean = df_a[
            df_a['_searchable_name'].notna() &
            df_a[self.provider_a.player_id_col].notna()
        ].copy()

        df_b_clean = df_b[
            df_b['_searchable_name'].notna() &
            df_b[self.provider_b.player_id_col].notna()
        ].copy()

        if len(df_a_clean) == 0 or len(df_b_clean) == 0:
            if verbose:
                print("Warning: One or both datasets are empty after filtering.")
            return pd.DataFrame()

        if verbose:
            print(f"Matching {len(df_a_clean)} records from {self.provider_a.name} "
                  f"with {len(df_b_clean)} records from {self.provider_b.name}")

        # Fit vectorizer on combined names
        all_names = pd.concat([
            df_a_clean['_searchable_name'],
            df_b_clean['_searchable_name']
        ])
        self.matcher.fit_vectorizer(all_names)

        # Match A -> B
        if verbose:
            print(f"\tMatching {self.provider_a.name} -> {self.provider_b.name}")
        matches_a_to_b = self.matcher.match_most_similar(
            master_name=df_b_clean['_searchable_name'],
            master_id=df_b_clean[self.provider_b.player_id_col],
            duplicates_name=df_a_clean['_searchable_name'],
            duplicates_id=df_a_clean[self.provider_a.player_id_col],
            verbose=verbose
        )

        # Match B -> A
        if verbose:
            print(f"\tMatching {self.provider_b.name} -> {self.provider_a.name}")
        matches_b_to_a = self.matcher.match_most_similar(
            master_name=df_a_clean['_searchable_name'],
            master_id=df_a_clean[self.provider_a.player_id_col],
            duplicates_name=df_b_clean['_searchable_name'],
            duplicates_id=df_b_clean[self.provider_b.player_id_col],
            verbose=verbose
        )

        # Find mutual matches
        mutual_matches = self._find_mutual_matches(matches_a_to_b, matches_b_to_a)

        if verbose:
            print(f"\tFound {len(mutual_matches)} mutual matches")

        # Add the ID of the other provider
        df_a[f'tmp_{self.provider_b.player_id_col}_{self.provider_b.name}'] = df_a[self.provider_a.player_id_col].map(mutual_matches)
        # Name the output columns explicitly. pandas only applies merge suffixes to
        # column names present in both frames, so providers with different id or
        # name columns (e.g. 'id' vs 'scoutastic_id') used to raise a KeyError here.
        a_id, a_name = self.provider_a.player_id_col, self.provider_a.player_name_col
        b_id, b_name = self.provider_b.player_id_col, self.provider_b.player_name_col
        left = df_a[[a_id, a_name, f'tmp_{b_id}_{self.provider_b.name}']].rename(columns={
            a_id: f'{a_id}_{self.provider_a.name}',
            a_name: f'{a_name}_{self.provider_a.name}',
        })
        right = df_b[[b_id, b_name]].rename(columns={
            b_id: f'{b_id}_{self.provider_b.name}',
            b_name: f'{b_name}_{self.provider_b.name}',
        })
        result = left.merge(
            right,
            left_on=f'tmp_{b_id}_{self.provider_b.name}',
            right_on=f'{b_id}_{self.provider_b.name}',
            how='left',
        )
        # only keep the columns of interest
        result = result[
            [
                f'{a_id}_{self.provider_a.name}',
                f'{a_name}_{self.provider_a.name}',
                f'{b_id}_{self.provider_b.name}',
                f'{b_name}_{self.provider_b.name}'
            ]
        ]

        return result

    def _find_mutual_matches(
        self,
        matches_a_to_b: pd.DataFrame,
        matches_b_to_a: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Find matches that are mutual (present in both directions).

        Args:
            matches_a_to_b: Matches from A to B
            matches_b_to_a: Matches from B to A

        Returns:
            DataFrame with mutual matches
        """
        # Create lookup dictionaries
        a_to_b = dict(zip(matches_a_to_b['duplicate_id'], matches_a_to_b['master_id']))
        b_to_a = dict(zip(matches_b_to_a['duplicate_id'], matches_b_to_a['master_id']))

        # Find mutual matches
        mutual = {key: value for key, value in a_to_b.items() if value in b_to_a and b_to_a[value] == key}

        return mutual

    def match_datasets(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Main method to match two datasets and optionally save results.

        Args:
            df_a: First dataset
            df_b: Second dataset
            output_file: Optional path to save matched results
            verbose: Whether to print progress messages

        Returns:
            DataFrame with matched records
        """
        # Prepare datasets
        df_a_prep = self.prepare_dataset(df_a, self.provider_a)
        df_b_prep = self.prepare_dataset(df_b, self.provider_b)

        # Find matches
        matches = self.find_bidirectional_matches(df_a_prep, df_b_prep, verbose=verbose)

        # Save if requested
        if output_file and len(matches) > 0:
            matches.to_csv(output_file, index=False)
            if verbose:
                print(f"\nResults saved to {output_file}")

        return matches


# Example usage
if __name__ == "__main__":
    # Configure providers
    soccerway_config = ProviderConfig(
        name='soccerway',
        player_name_col='player_name',
        player_id_col='player_id',
        date_of_birth_col='date_of_birth'
    )

    transfermarkt_config = ProviderConfig(
        name='transfermarkt',
        player_name_col='player_name',
        player_id_col='player_id',
        date_of_birth_col='date_of_birth'
    )

    # Load datasets
    soccerway_df = pd.read_csv('data/soccerway_dataset.csv', dtype={"player_id": str})
    transfermarkt_df = pd.read_csv('data/transfermarkt_dataset.csv', dtype={"player_id": str})

    # Initialize matcher
    matcher = BidirectionalPlayerMatcher(
        provider_a=soccerway_config,
        provider_b=transfermarkt_config,
        min_similarity=0.5,
        ngram_size=3
    )

    # Perform matching
    matched_players = matcher.match_datasets(
        df_a=soccerway_df,
        df_b=transfermarkt_df,
        output_file='results/matched_players.csv',
        verbose=True
    )

    print("\nSample matches:")
    print(matched_players.head())
