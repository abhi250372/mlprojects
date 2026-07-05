import pandas as pd
from pathlib import Path
from components.input_processor import InputProcessor
from components.input_transformer import InputTransformer


def main():
    input_processor = InputProcessor()
    input_transformer = InputTransformer()
    profile_data = input_processor.load_data('src/football_datasets/player_profiles/player_profiles.csv')
    player_performances = input_processor.load_data('src/football_datasets/player_performances/player_performances.csv')
    player_latest_market_values = input_processor.load_data('src/football_datasets/player_latest_market_value/player_latest_market_value.csv')
    player_latest_market_values = player_latest_market_values[player_latest_market_values['value'] > 0]
    position = profile_data['main_position'].value_counts()
    print('Profile columns:', profile_data.columns.tolist())
    print(position)
    print('Player performances columns:', player_performances.columns.tolist())
    print(player_performances.head())
    print('Player latest market values columns:', player_latest_market_values.columns.tolist())
    print(player_latest_market_values.shape[0], 'players with market values')

    current_market_value_player_data = profile_data.merge(player_latest_market_values, on='player_id', how='inner')
    print('Current market value player data shape:', current_market_value_player_data.shape)

    numeric_performance_metrics = player_performances.select_dtypes(include='number').columns.tolist()
    numeric_performance_metrics = [c for c in numeric_performance_metrics if c not in ['player_id', 'competition_id', 'team_id']]

    performance_by_season = input_transformer.aggregate_player_performances(
        player_performances,
        season_col='season_name',
        player_col='player_id',
        metric_cols=numeric_performance_metrics,
    )
    print('Player performance by season shape:', performance_by_season.shape)
    print(performance_by_season.head())

    weighted_performance = input_transformer.weighted_season_average(
        performance_by_season,
        player_col='player_id',
        season_col='season_name',
        metric_cols=[c for c in performance_by_season.columns if c not in ['player_id', 'season_name']],
    )
    print('Weighted performance columns:', weighted_performance.columns.tolist())
    print(weighted_performance.head())

    current_market_value_player_data = current_market_value_player_data.merge(weighted_performance, on='player_id', how='inner')
    current_market_value_player_data = current_market_value_player_data[['player_id', 'player_name', 'main_position', 'position', 'value'] + [c for c in weighted_performance.columns if c not in ['player_id']]]
    print('Current market value player data shape:', current_market_value_player_data.shape)
    print('Current market value player data columns:', current_market_value_player_data.columns.tolist())

    output_path = Path('src/football_datasets/current_market_value_player_data.csv')
    current_market_value_player_data.to_csv(output_path, index=False)
    print('Exported merged player data to', output_path)


if __name__ == "__main__":
    main()