import pandas as pd

input_path = '../input'
competition_path = f'{input_path}/indoor-location-navigation'
submissions_path = f'{input_path}/submissions'

sample_submission_path = f'{competition_path}/sample_submission.csv'

submission_1_path = f'{submissions_path}/submission_lstm_x_y_with_floor.csv'
submission_2_path = f'{submissions_path}/submission_lstm_floor_pred_100_feats.csv'

sample_subm = pd.read_csv(sample_submission_path,
                          index_col=0)

subm1 = pd.read_csv(
    submission_1_path,
    dtype={"site_path_timestamp": str, "floor": int},
    usecols=['site_path_timestamp', 'floor', 'x', 'y'],
    index_col=0
)

subm2 = pd.read_csv(
    submission_2_path,
    dtype={"site_path_timestamp": str, "floor": int, },
    usecols=['site_path_timestamp', 'floor', 'x', 'y'],
    index_col=0
)

site_path_timestamps = subm1.index

amount = 0
for site_path_timestamp in site_path_timestamps:
    timestamp = subm1[subm1.index == site_path_timestamp]
    timestamp2 = subm2[subm2.index == site_path_timestamp]
    floor_no = timestamp["floor"].iloc[0]
    floor_no2 = timestamp2["floor"].iloc[0]
    if floor_no != floor_no2:
        amount += 1
        print(f"{amount}. {site_path_timestamp} First: {floor_no}, Second {floor_no2}")

percentage = amount / len(site_path_timestamps) * 100

print(f'difference: {amount}, total percentage: {percentage}')