$TASK = "word_onset"
$IDX  = "idx07"

cd "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer"

python -m data.write_multi_subject_multi_channel `
  "+data_prep=pretrain_multi_subj_multi_chan_template" `
  "++data_prep.task_name=$TASK" `
  "++data_prep.brain_runs=$REPO_DIR/trial_selections/speech_sub1_2_word_onset.json" `
  "++data_prep.electrodes=$REPO_DIR/electrode_selections/clean_laplacian.json" `
  "++data_prep.output_directory=$REPO_DIR/saved_examples/all_test_${TASK}_$IDX" `
  "++data_prep.index_subsample=1.0" `
  "++data_prep.separation_interval=0.5" `
  "+preprocessor=multi_elec_spec_pretrained" `
  "++preprocessor.upstream_ckpt=$STFT_CKPT" `
  "+data=subject_data_template" `
  "++data.cached_transcript_aligns=$CACHED_ALIGNS" `
  "++data.cached_data_array=$CACHED_ARRAYS" `
  "++data.raw_brain_data_dir=$BRAINTREEBANK_DIR" `
  "++data.movie_transcripts_dir=$TRANSCRIPTS_DIR" `
  "++data.rereference=None" `
  "++data.duration=1.0" `
  "++data.interval_duration=1.0"

saved_examples/all_test_word_onset_idx07/sub_1/...
saved_examples/all_test_word_onset_idx07/sub_2/...

python -m data.make_subject_specific_manifest `
  "+data_prep=subject_specific_manifest" `
  "++data_prep.data_path=$REPO_DIR/saved_examples/all_test_${TASK}_$IDX" `
  "++data_prep.subj=sub_1" `
  "++data_prep.out_path=$REPO_DIR/saved_examples/sub_1_${TASK}_cr_$IDX"

python -m data.make_subject_specific_manifest `
  "+data_prep=subject_specific_manifest" `
  "++data_prep.data_path=$REPO_DIR/saved_examples/all_test_${TASK}_$IDX" `
  "++data_prep.subj=sub_2" `
  "++data_prep.out_path=$REPO_DIR/saved_examples/sub_2_${TASK}_cr_$IDX"

saved_examples/sub_1_word_onset_cr_idx07/manifest.tsv, labels.tsv, ...
saved_examples/sub_2_word_onset_cr_idx07/manifest.tsv, labels.tsv, ...

$TASK            = "word_onset"
$IDX             = "idx07"
$SUBJECT         = "sub_1"
$N               = 1
$NAME            = "popt_brainbert_stft_${TASK}_$IDX"
$PRETRAINED_POPT = "$BRAIN_REPO\pretrained_weights\popt_pretrained_weights\pretrained_popt_brainbert_stft.pth"
$RESULTS_DIR     = "$REPO_DIR\outputs\${SUBJECT}_${TASK}_top${N}_${NAME}"

python run_train.py `
  "+exp=multi_elec_feature_extract" `
  "++exp.runner.results_dir=$RESULTS_DIR" `
  "++exp.runner.save_checkpoints=False" `
  "++model.frozen_upstream=True" `   # linear probe run
  "+task=pt_feature_extract_coords" `
  "+criterion=pt_feature_extract_coords_criterion" `
  "+preprocessor=empty_preprocessor" `
  "+data=pt_supervised_task_coords" `
  "++data.data_path=$REPO_DIR/saved_examples/${SUBJECT}_${TASK}_cr_$IDX" `
  "++data.saved_data_split=$REPO_DIR/saved_data_splits/${SUBJECT}_${TASK}_fine_tuning_$IDX" `
  "++data.sub_sample_electrodes=$REPO_DIR/electrode_selections/clean_laplacian.json" `
  "+model=pt_downstream_model" `
  "++model.upstream_path=$PRETRAINED_POPT"

If you also want a full fine-tune version, rerun with:

"++model.frozen_upstream=False"

$TASK            = "word_onset"
$IDX             = "idx07"
$SUBJECT         = "sub_2"
$N               = 1
$NAME            = "popt_brainbert_stft_${TASK}_$IDX"
$PRETRAINED_POPT = "$BRAIN_REPO\pretrained_weights\popt_pretrained_weights\pretrained_popt_brainbert_stft.pth"
$RESULTS_DIR     = "$REPO_DIR\outputs\${SUBJECT}_${TASK}_top${N}_${NAME}"

python run_train.py `
  "+exp=multi_elec_feature_extract" `
  "++exp.runner.results_dir=$RESULTS_DIR" `
  "++exp.runner.save_checkpoints=False" `
  "++model.frozen_upstream=True" `
  "+task=pt_feature_extract_coords" `
  "+criterion=pt_feature_extract_coords_criterion" `
  "+preprocessor=empty_preprocessor" `
  "+data=pt_supervised_task_coords" `
  "++data.data_path=$REPO_DIR/saved_examples/${SUBJECT}_${TASK}_cr_$IDX" `
  "++data.saved_data_split=$REPO_DIR/saved_data_splits/${SUBJECT}_${TASK}_fine_tuning_$IDX" `
  "++data.sub_sample_electrodes=$REPO_DIR/electrode_selections/clean_laplacian.json" `
  "+model=pt_downstream_model" `
  "++model.upstream_path=$PRETRAINED_POPT"

  (Optionally repeat with frozen_upstream=False for a fine-tuned run.)

  outputs\sub_1_word_onset_top1_popt_brainbert_stft_word_onset_idx07\results.json
outputs\sub_2_word_onset_top1_popt_brainbert_stft_word_onset_idx07\results.json
Re-use your existing Python analysis script, just change results_path to point to each of those files in turn, and save the plots or print out:

ROC AUC

best F1 + threshold

precision/recall snapshot


