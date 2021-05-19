import pandas as pd

#train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
#test_file = pd.read_csv(f'{feature_dir}/test.csv')

#all_train_df = None
#
#for train_file in train_files:
#    train_df = pd.read_csv(train_file)
#    if all_train_df is None:
#        all_train_df = train_df
#    else:
#        all_train_df = pd.concat([all_train_df, train_df], axis=0)
#
#with open(f'{feature_dir}/train.pkl', 'wb') as f:
#    pickle.dump(all_train_df, f)
#
#with open(f'{feature_dir}/test.pkl', 'wb') as f:
#    pickle.dump(test_file, f)
#
#exit()