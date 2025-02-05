import numpy as np
import pandas as pd

def analyze_equity_curve(equity_curve: list) -> dict:
    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    segments = []
    current_segment = {'start': df.loc[0, 'date'], 'start_value': df.loc[0, 'portfolio_value']}
    current_sign = np.sign(df.loc[1, 'portfolio_value'] - df.loc[0, 'portfolio_value'])
    for i in range(1, len(df)):
        change = df.loc[i, 'portfolio_value'] - df.loc[i-1, 'portfolio_value']
        sign = np.sign(change)
        if sign == 0:
            sign = current_sign
        if sign != current_sign:
            end_date = df.loc[i-1, 'date']
            end_value = df.loc[i-1, 'portfolio_value']
            pct_change = (end_value - current_segment['start_value']) / current_segment['start_value']
            segments.append({
                'start': current_segment['start'].strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'pct_change': pct_change
            })
            current_segment = {'start': df.loc[i, 'date'], 'start_value': df.loc[i, 'portfolio_value']}
            current_sign = sign
    end_date = df.loc[len(df)-1, 'date']
    end_value = df.loc[len(df)-1, 'portfolio_value']
    pct_change = (end_value - current_segment['start_value']) / current_segment['start_value']
    segments.append({
        'start': current_segment['start'].strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d'),
        'pct_change': pct_change
    })
    good_periods = [seg for seg in segments if seg['pct_change'] > 0]
    poor_periods = [seg for seg in segments if seg['pct_change'] < 0]
    return {'good_periods': good_periods, 'poor_periods': poor_periods}

def analyze_training_history(history: dict) -> dict:
    val_loss = history.get('val_loss', [])
    if not val_loss:
        return {}
    best_epoch = int(np.argmin(val_loss) + 1)
    worst_epoch = int(np.argmax(val_loss) + 1)
    return {
        'best_epoch': best_epoch,
        'best_val_loss': val_loss[best_epoch - 1],
        'worst_epoch': worst_epoch,
        'worst_val_loss': val_loss[worst_epoch - 1]
    }
