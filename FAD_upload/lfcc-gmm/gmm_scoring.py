from gmm import scoring


# scores file to write
scores_file = 'FAD_clean/scores-test-unseen.txt'

# configs
features = 'lfcc'
dict_file = 'FAD_clean/gmm_lfcc.pkl'

db_folder ='./FAD/clean/'  # put your database root path here
eval_folder = db_folder +'test'
eval_ndx = db_folder + 'test/test_unseen_gmm_label.txt' 

audio_ext = '.wav'

# run on test set
scoring(scores_file=scores_file, dict_file=dict_file, features=features,
        eval_ndx=eval_ndx, eval_folder=eval_folder, audio_ext=audio_ext,
        features_cached=False) #True)
