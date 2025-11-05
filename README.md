# Player Matcher

A production-ready fuzzy matching framework for linking player records across multiple data providers using TF-IDF vectorization and bidirectional validation.

More information can be found on [Gialloblu Open Analytics Lab](https://pc1913-perf-analytics.notion.site/Matching-players-across-different-providers-294a95fcfc3280638509dfc3b0e8b8b6), our blog.

## Installation

### Prerequisites

- Python 3.11
- pip

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/parmacalcio1913/players-matcher.git
cd players-matcher
```

2. **Create a virtual environment and install the packages**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start
Simply run:

```bash
python main.py
```

This will:
1. Load sample datasets from Soccerway and Transfermarkt
2. Perform bidirectional fuzzy matching
3. Output results to `matched_players.csv`

## Expected Data Format

Your input CSV files should contain at minimum:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `player_name` | string | Player's full name | Yes |
| `player_id` | string/int | Unique identifier | Yes |
| `date_of_birth` | string | Date in `YYYY-MM-DD` format | Yes |

Example:

```csv
player_name,player_id,date_of_birth
Lionel Messi,12345,1987-06-24
Cristiano Ronaldo,67890,1985-02-05
```

## Extending the Solution

To add support for a new data source, simply create a new `ProviderConfig`:

```python
# Example: Adding FBref support
fbref_config = ProviderConfig(
    name='fbref',
    player_name_col='Player',           # Column name in your CSV
    player_id_col='Player_ID',          # Column name in your CSV
    date_of_birth_col='Birth_Date'      # Column name in your CSV
)

# Use it in matching
matcher = BidirectionalPlayerMatcher(
    provider_a=soccerway_config,
    provider_b=fbref_config,
    min_similarity=0.5,
    ngram_size=3
)
```


**Parameter Guidelines:**

- `min_similarity`:
  - `0.3-0.4`: Very permissive, may produce false positives
  - `0.5-0.6`: Balanced (recommended for most use cases)
  - `0.7-0.8`: Conservative, may miss some valid matches

- `ngram_size`:
  - `2`: Very fuzzy, good for short names or heavy typos
  - `3`: Balanced (recommended default)
  - `4-5`: More strict, better for longer names

## Algorithm Details
For all the implementation details, visit our [blog](https://pc1913-perf-analytics.notion.site/Matching-players-across-different-providers-294a95fcfc3280638509dfc3b0e8b8b6).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments
Our solution is largely based on:
- [string_grouper](https://github.com/Bergvca/string_grouper) by Bergvca
- [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn) by ING Bank
