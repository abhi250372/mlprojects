import pandas as pd


class InputTransformer:
    def __init__(self):
        self.data = None

    def _season_order_key(self, season_value):
        if pd.isna(season_value):
            return float('-inf')
        value = str(season_value).strip()
        if '/' in value:
            start = value.split('/')[0]
            if len(start) == 2 and start.isdigit():
                year = int(start)
                return year + (2000 if year < 70 else 1900)
            if start.isdigit():
                return int(start)
        try:
            return int(value)
        except ValueError:
            return value

    def aggregate_player_performances(self, player_performances_df, season_col='season_name', player_col='player_id', metric_cols=None):
        if not isinstance(player_performances_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if metric_cols is None:
            metric_cols = player_performances_df.select_dtypes(include='number').columns.tolist()
            metric_cols = [c for c in metric_cols if c not in [player_col, 'competition_id', 'team_id']]

        grouped = player_performances_df.groupby([player_col, season_col], as_index=False)[metric_cols].sum()
        return grouped

    def weighted_season_average(self, performance_by_season_df, player_col='player_id', season_col='season_name', metric_cols=None):
        if not isinstance(performance_by_season_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if metric_cols is None:
            metric_cols = [c for c in performance_by_season_df.columns if c not in [player_col, season_col]]

        df = performance_by_season_df.copy()
        df['season_order'] = df[season_col].apply(self._season_order_key)
        df = df.sort_values([player_col, 'season_order'], ascending=[True, True])
        df['season_rank'] = df.groupby(player_col).cumcount() + 1
        df['weight'] = df['season_rank']

        weighted_sum = df.copy()
        weighted_sum[metric_cols] = weighted_sum[metric_cols].multiply(weighted_sum['weight'], axis=0)
        numerator = weighted_sum.groupby(player_col)[metric_cols].sum()
        denominator = df.groupby(player_col)['weight'].sum()

        weighted = numerator.div(denominator, axis=0).fillna(0).reset_index()
        return weighted

    def transform_player_performances(self, player_performances_df, season_col='season_name', player_col='player_id', metric_cols=None):
        if not isinstance(player_performances_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        season_stats = self.aggregate_player_performances(
            player_performances_df,
            season_col=season_col,
            player_col=player_col,
            metric_cols=metric_cols,
        )

        weighted_features = self.weighted_season_average(
            season_stats,
            player_col=player_col,
            season_col=season_col,
            metric_cols=[c for c in season_stats.columns if c not in [player_col, season_col]],
        )

        self.data = weighted_features
        return self.data