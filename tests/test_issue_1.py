"""
Regression tests for issue #1: providers whose id or name columns differ.

Run from the repository root with:

    python -m pytest tests
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main import BidirectionalPlayerMatcher, ProviderConfig  # noqa: E402

# sha256 of results/matched_players.csv written by main.py at a643af2 on the sample data (LF line endings)
SAMPLE_OUTPUT_SHA256 = '050805c93362829c85f713a6e13a458e06b15327fd3705cc820e686245c9878a'


def _issue_1_frames():
    """The provider setup from issue #1: SkillCorner uses 'id' and 'birthdate', Scoutastic uses 'scoutastic_id'."""
    skillcorner = pd.DataFrame({
        'id': ['1', '2', '3'],
        'player_name': ['Lionel Messi', 'Kylian Mbappe', 'Erling Haaland'],
        'birthdate': ['1987-06-24', '1998-12-20', '2000-07-21'],
    })
    scoutastic = pd.DataFrame({
        'scoutastic_id': ['x', 'y', 'z'],
        'player_name': ['Lionel Andres Messi', 'Kylian Mbappé', 'Erling Braut Haaland'],
        'date_of_birth': ['1987-06-24', '1998-12-20', '2000-07-21'],
    })
    return skillcorner, scoutastic


def test_different_id_columns():
    skillcorner, scoutastic = _issue_1_frames()
    matcher = BidirectionalPlayerMatcher(
        provider_a=ProviderConfig(name='skillcorner', player_name_col='player_name',
                                  player_id_col='id', date_of_birth_col='birthdate'),
        provider_b=ProviderConfig(name='scoutastic', player_name_col='player_name',
                                  player_id_col='scoutastic_id', date_of_birth_col='date_of_birth'),
        min_similarity=0.5,
        ngram_size=3
    )

    result = matcher.match_datasets(skillcorner, scoutastic, verbose=False)

    assert list(result.columns) == [
        'id_skillcorner', 'player_name_skillcorner', 'scoutastic_id_scoutastic', 'player_name_scoutastic'
    ]
    assert list(zip(result['id_skillcorner'], result['scoutastic_id_scoutastic'])) == [
        ('1', 'x'), ('2', 'y'), ('3', 'z')
    ]


def test_different_name_columns():
    skillcorner, scoutastic = _issue_1_frames()
    scoutastic = scoutastic.rename(columns={'player_name': 'Player'})
    matcher = BidirectionalPlayerMatcher(
        provider_a=ProviderConfig(name='skillcorner', player_id_col='id', date_of_birth_col='birthdate'),
        provider_b=ProviderConfig(name='scoutastic', player_name_col='Player', player_id_col='scoutastic_id'),
    )

    result = matcher.match_datasets(skillcorner, scoutastic, verbose=False)

    assert list(result.columns) == [
        'id_skillcorner', 'player_name_skillcorner', 'scoutastic_id_scoutastic', 'Player_scoutastic'
    ]
    assert result['Player_scoutastic'].tolist() == scoutastic['Player'].tolist()


def test_same_provider_name_raises_value_error():
    with pytest.raises(ValueError, match="different names"):
        BidirectionalPlayerMatcher(provider_a=ProviderConfig(name='opta'), provider_b=ProviderConfig(name='opta'))


def test_sample_output_unchanged(tmp_path):
    soccerway = pd.read_csv(REPO_ROOT / 'data' / 'soccerway_dataset.csv', dtype={"player_id": str})
    transfermarkt = pd.read_csv(REPO_ROOT / 'data' / 'transfermarkt_dataset.csv', dtype={"player_id": str})
    output = tmp_path / 'matched_players.csv'

    BidirectionalPlayerMatcher(
        provider_a=ProviderConfig(name='soccerway'),
        provider_b=ProviderConfig(name='transfermarkt'),
        min_similarity=0.5,
        ngram_size=3
    ).match_datasets(soccerway, transfermarkt, output_file=str(output), verbose=False)

    # pandas writes os.linesep, so compare with LF line endings
    content = output.read_bytes().replace(bytes([13, 10]), bytes([10]))
    assert hashlib.sha256(content).hexdigest() == SAMPLE_OUTPUT_SHA256
