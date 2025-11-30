Task list 

rms, pitch, pos, rms_reg, word_onset, sentence_onset, pretraining, spec_target_pretraining

$REPO_DIR  = "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer"
$BRAIN_REPO = "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\Brain-Treebank"
$BRAINTREEBANK_DIR = "$BRAIN_REPO\braintreebank_data"

$STFT_CKPT     = "$BRAIN_REPO\pretrained_weights\brainbert_pretrained_weights\stft_large_pretrained.pth"
$PRETRAINED_POPT = "$BRAIN_REPO\pretrained_weights\popt_pretrained_weights\pretrained_popt_brainbert_stft.pth"

$RESULTS_DIR = "$REPO_DIR\outputs\sub_1_word_onset_top1_popt_brainbert_stft"

$TASK = "word_onset" 
$TASK = "sentence_onset"


# Restrict to sub_1, trials 000 and 002

$trialJson = @"
{
  "sub_1": ["trial000", "trial002"]
}
"@

New-Item -ItemType Directory -Force -Path "$REPO_DIR\trial_selections" | Out-Null
$trialJson | Out-File -Encoding utf8 "$REPO_DIR\trial_selections\speech_sub1_000_002.json"

# Build the supervised dataset for word_onset
$CACHED_ALIGNS   = "$BRAIN_REPO\semantics\saved_aligns"
$CACHED_ARRAYS   = "$BRAIN_REPO\cached_data_arrays"
$TRANSCRIPTS_DIR = "$BRAINTREEBANK_DIR\transcripts"



# write_multi_subject_multi_channel SCRIPT

''''
(dl_gpu_env) PS C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer> $REPO_DIR = "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer"
(dl_gpu_env) PS C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer> python -c "import json, pathlib; p = pathlib.Path(r'$REPO_DIR\trial_selections\speech_sub1_000_002.json'); p.write_text(json.dumps({'sub_1':['trial000','trial002']}), encoding='utf-8')"
(dl_gpu_env) PS C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer> Get-Content "$REPO_DIR\trial_selections\speech_sub1_000_002.json"
{"sub_1": ["trial000", "trial002"]}

'''

$TASK = "sentence_onset"
$IDX  = "idx05"

python -m data.write_multi_subject_multi_channel `
  "+data_prep=pretrain_multi_subj_multi_chan_template" `
  "++data_prep.task_name=$TASK" `
  "++data_prep.brain_runs=$REPO_DIR/trial_selections/speech_sub1_000_002.json" `
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

$TASK = "word_onset"
$IDX  = "idx06" and "idx07,speech_sub1_2_word_onset"
  
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

# solution 
Turn off rereference ✅ (already did, rereference=None)
Shrink the window duration (5s → 1s)
Subsample the number of intervals (e.g., use ~10% of them

What this does:
++data.duration=1.0 → windows 1 second long instead of 5s
++data_prep.index_subsample=0.1 → use ~10% of intervals

Keeps the same word-onset logic (same labels, same pipeline).
Uses shorter 1s windows → 5× less memory per window.
Uses 10% of all intervals → 10× fewer windows.
Combined memory reduction ≈ 50× vs the original 51 GB monster → into the ~1 GB range, which is realistic.


# MANIFEST
SUB 1

python -m data.make_subject_specific_manifest `
  "+data_prep=subject_specific_manifest" `
  "++data_prep.data_path=$REPO_DIR/saved_examples/all_test_${TASK}_$IDX" `
  "++data_prep.subj=sub_1" `
  "++data_prep.out_path=$REPO_DIR/saved_examples/sub_1_${TASK}_cr_$IDX"


# FINE TUNING

$TASK           = "sentence_onset"
$IDX            = "idx05"
$SUBJECT        = "sub_1"
$N              = 1
$NAME           = "popt_brainbert_stft_${TASK}_$IDX"
$PRETRAINED_POPT = "$BRAIN_REPO\pretrained_weights\popt_pretrained_weights\pretrained_popt_brainbert_stft.pth"
$RESULTS_DIR    = "$REPO_DIR\outputs\${SUBJECT}_${TASK}_top${N}_${NAME}"

python run_train.py `
  "+exp=multi_elec_feature_extract" `
  "++exp.runner.results_dir=$RESULTS_DIR" `
  "++exp.runner.save_checkpoints=False" `
  "++model.frozen_upstream=True" `
  "+task=pt_feature_extract_coords" `
  "+criterion=pt_feature_extract_coords_criterion" `
  "+preprocessor=empty_preprocessor" `
  "+data=pt_supervised_task_coords" `
  "++data.data_path=$REPO_DIR/saved_examples/sub_1_${TASK}_cr_$IDX" `
  "++data.saved_data_split=$REPO_DIR/saved_data_splits/sub_1_${TASK}_fine_tuning_$IDX" `
  "++data.sub_sample_electrodes=$REPO_DIR/electrode_selections/clean_laplacian.json" `
  "+model=pt_downstream_model" `
  "++model.upstream_path=$PRETRAINED_POPT"

# METIRCS

Get-Content "$RESULTS_DIR\results.json"