import json
from collections import defaultdict
import pandas as pd

class InputProcessor:
    def __init__(self):
        self.data = None

    def load_data(self, file_path=None):
        self.data = pd.read_csv(file_path)
        return self.data

    def load_json(self, file_path=None):
        with open(file_path, encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data

    def load_statsbomb_events(self, file_path=None):
        events = self.load_json(file_path)
        if isinstance(events, list):
            return pd.DataFrame(events)
        return events

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

    def aggregate_player_performances(self, df, season_col='season_name', player_col='player_id', metric_cols=None):
        if metric_cols is None:
            metric_cols = df.select_dtypes(include='number').columns.tolist()
            metric_cols = [c for c in metric_cols if c != player_col]

        grouped = df.groupby([player_col, season_col], as_index=False)[metric_cols].sum()
        return grouped

    def weighted_season_average(self, df, player_col='player_id', season_col='season_name', metric_cols=None):
        if metric_cols is None:
            metric_cols = [c for c in df.columns if c not in [player_col, season_col]]

        df = df.copy()
        df['season_order'] = df[season_col].apply(self._season_order_key)
        df = df.sort_values([player_col, 'season_order'], ascending=[True, True])
        df['season_rank'] = df.groupby(player_col).cumcount() + 1
        df['weight'] = df['season_rank']

        weighted = df.groupby(player_col).apply(
            lambda g: pd.Series({
                col: (g[col] * g['weight']).sum() / g['weight'].sum() if g['weight'].sum() else 0
                for col in metric_cols
            })
        ).reset_index()

        return weighted

    def aggregate_statsbomb_events(self, file_path=None, events_df=None, first_n=None):
        if events_df is None:
            events = self.load_json(file_path)
        elif isinstance(events_df, pd.DataFrame):
            events = events_df.to_dict(orient='records')
        else:
            events = events_df

        if first_n is not None and isinstance(first_n, int):
            events = events[:first_n]

        player_stats = defaultdict(lambda: defaultdict(int))

        def is_progressive_pass(ev):
            if ev.get('type', {}).get('name') != 'Pass':
                return False
            p = ev.get('pass') or {}
            if not ev.get('location') or not p.get('end_location'):
                return False
            start_x = ev['location'][0]
            end_x = p['end_location'][0]
            return end_x > start_x + 10

        def is_progressive_carry(ev):
            if ev.get('type', {}).get('name') != 'Carry':
                return False
            c = ev.get('carry') or {}
            if not ev.get('location') or not c.get('end_location'):
                return False
            start_x = ev['location'][0]
            end_x = c['end_location'][0]
            return end_x > start_x + 10

        for ev in events:
            player = ev.get('player')
            if not player:
                continue
            pid = player.get('id')
            if pid is None:
                continue

            stats = player_stats[pid]
            stats['player_id'] = pid
            stats['player_name'] = player.get('name')

            etype = ev.get('type', {}).get('name')

            if etype == 'Pass':
                stats['passes'] += 1
                if ev.get('pass', {}).get('outcome') is None:
                    stats['pass_completions'] += 1
                if is_progressive_pass(ev):
                    stats['progressive_passes'] += 1

            elif etype == 'Carry':
                stats['carries'] += 1
                if is_progressive_carry(ev):
                    stats['progressive_carries'] += 1

            elif etype == 'Interception':
                stats['interceptions'] += 1

            elif etype == 'Ball Recovery':
                stats['ball_recoveries'] += 1

            elif etype == 'Shot':
                stats['shots'] += 1
                if ev.get('shot', {}).get('outcome', {}).get('name') == 'Goal':
                    stats['goals'] += 1

            elif etype == 'Duel':
                stats['duels'] += 1
                if ev.get('outcome', {}).get('name') == 'Success':
                    stats['duel_successes'] += 1

        df = pd.DataFrame(player_stats.values())
        return df.fillna(0)