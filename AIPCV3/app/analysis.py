import numpy as np
import pandas as pd

def analyze_equity_curve(equity_curve: list) -> dict:
    # we convert the equity_curve list into a pandas dataframe
    df = pd.DataFrame(equity_curve)
    # we convert the 'date' column to a datetime type for proper sorting
    df['date'] = pd.to_datetime(df['date'])
    # we sort the dataframe by date to ensure correct chronological order
    df = df.sort_values('date').reset_index(drop=True)

    # we prepare to identify segments of growth or decline in the portfolio_value
    segments = []
    current_segment = {'start': df.loc[0, 'date'], 'start_value': df.loc[0, 'portfolio_value']}
    # we check the sign of the portfolio change between the first two points
    current_sign = np.sign(df.loc[1, 'portfolio_value'] - df.loc[0, 'portfolio_value'])
    for i in range(1, len(df)):
        # we compute the daily change
        change = df.loc[i, 'portfolio_value'] - df.loc[i-1, 'portfolio_value']
        # we find the sign of that change (positive or negative)
        sign = np.sign(change)
        # if sign is zero (no change), maintain the current sign
        if sign == 0:
            sign = current_sign
        # if the sign changes, we end the current segment and start a new one
        if sign != current_sign:
            end_date = df.loc[i-1, 'date']
            end_value = df.loc[i-1, 'portfolio_value']
            pct_change = (end_value - current_segment['start_value']) / current_segment['start_value']
            # we store that finished segment in the list
            segments.append({
                'start': current_segment['start'].strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'pct_change': pct_change
            })
            # we initialize a new segment starting at the current index
            current_segment = {'start': df.loc[i, 'date'], 'start_value': df.loc[i, 'portfolio_value']}
            current_sign = sign

    # we close out the final segment after looping
    end_date = df.loc[len(df)-1, 'date']
    end_value = df.loc[len(df)-1, 'portfolio_value']
    pct_change = (end_value - current_segment['start_value']) / current_segment['start_value']
    segments.append({
        'start': current_segment['start'].strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d'),
        'pct_change': pct_change
    })

    # we separate segments into good (positive pct_change) and poor (negative pct_change) periods
    good_periods = [seg for seg in segments if seg['pct_change'] > 0]
    poor_periods = [seg for seg in segments if seg['pct_change'] < 0]
    # we return both good and poor periods for further analysis
    return {'good_periods': good_periods, 'poor_periods': poor_periods}

def analyze_training_history(history: dict) -> dict:
    # we fetch the validation losses from the training history
    val_loss = history.get('val_loss', [])
    # if there is no validation loss, return an empty dict
    if not val_loss:
        return {}
    # we find the epoch with the lowest loss (best epoch) and highest loss (worst epoch)
    best_epoch = int(np.argmin(val_loss) + 1)
    worst_epoch = int(np.argmax(val_loss) + 1)
    # we return a summary of the best and worst epochs and their corresponding val losses
    return {
        'best_epoch': best_epoch,
        'best_val_loss': val_loss[best_epoch - 1],
        'worst_epoch': worst_epoch,
        'worst_val_loss': val_loss[worst_epoch - 1]
    }
