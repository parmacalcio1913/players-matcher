workspace "Player Matcher" "A fuzzy matching framework for linking player records across multiple data providers using TF-IDF vectorization and bidirectional validation." {

    model {
        analyst = person "Data Analyst" "Runs the matching pipeline to link player records across providers"

        playerMatcher = softwareSystem "Player Matcher" "Matches player records from different data providers using TF-IDF n-gram similarity and bidirectional validation" {
            cli = container "CLI Entry Point" "Loads CSV datasets, configures providers, orchestrates matching, writes output" "Python, main.py"
            nameProcessor = container "Player Name Processor" "Cleans names (unidecode, lowercase, strip punctuation) and encodes dates of birth into compact tokens" "Python, PlayerNameProcessor"
            tfidfMatcher = container "TF-IDF Matcher" "Vectorizes searchable names into TF-IDF n-gram matrices and computes sparse top-N similarity" "Python, TfidfMatcher"
            bidirectionalMatcher = container "Bidirectional Matcher" "Orchestrates A-to-B and B-to-A matching, then filters to mutual matches only" "Python, BidirectionalPlayerMatcher"
        }

        soccerway = softwareSystem "Soccerway Dataset" "CSV file with player records from Soccerway (name, DOB, country)" "External"
        transfermarkt = softwareSystem "Transfermarkt Dataset" "CSV file with player records from Transfermarkt (name, DOB)" "External"
        outputCsv = softwareSystem "Matched Output" "CSV file containing bidirectionally validated player ID mappings" "External"

        analyst -> cli "Runs" "python main.py"
        cli -> soccerway "Reads player records from" "pandas read_csv"
        cli -> transfermarkt "Reads player records from" "pandas read_csv"
        cli -> bidirectionalMatcher "Delegates matching to"
        bidirectionalMatcher -> nameProcessor "Cleans names and encodes DOBs via"
        bidirectionalMatcher -> tfidfMatcher "Computes similarity via"
        cli -> outputCsv "Writes matched pairs to" "pandas to_csv"
    }

    views {
        systemContext playerMatcher "SystemContext" {
            include *
            autoLayout
        }

        container playerMatcher "Containers" {
            include *
            autoLayout
        }

        dynamic playerMatcher "MatchingFlow" {
            analyst -> cli "Execute matching pipeline"
            cli -> soccerway "Load provider A records"
            cli -> transfermarkt "Load provider B records"
            cli -> bidirectionalMatcher "Match datasets"
            bidirectionalMatcher -> nameProcessor "Clean names and encode DOBs"
            bidirectionalMatcher -> tfidfMatcher "Fit TF-IDF vectorizer on combined names"
            bidirectionalMatcher -> tfidfMatcher "Match A to B (sparse top-N)"
            bidirectionalMatcher -> tfidfMatcher "Match B to A (sparse top-N)"
            bidirectionalMatcher -> cli "Return mutual matches"
            cli -> outputCsv "Write matched_players.csv"
            autoLayout
        }

        styles {
            element "Person" {
                shape Person
                background #08427B
                color #ffffff
            }
            element "Software System" {
                background #1168BD
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Container" {
                background #438DD5
                color #ffffff
            }
            element "Component" {
                background #85BBF0
                color #000000
            }
        }
    }

}
