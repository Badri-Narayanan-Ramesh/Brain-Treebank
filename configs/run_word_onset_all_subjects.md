# ============================
# Word Onset – All 5 Subjects
# ============================

# --- Base paths ---
$REPO_DIR   = "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer"
$BRAIN_REPO = "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\Brain-Treebank"
$BRAINTREEBANK_DIR = "$BRAIN_REPO\braintreebank_data"

# --- Pretrained checkpoints ---
$STFT_CKPT       = "$BRAIN_REPO\pretrained_weights\brainbert_pretrained_weights\stft_large_pretrained.pth"
$PRETRAINED_POPT = "$BRAIN_REPO\pretrained_weights\popt_pretrained_weights\pretrained_popt_brainbert_stft.pth"

# --- Cached resources & transcripts ---
$CACHED_ALIGNS   = "$BRAIN_REPO\semantics\saved_aligns"
$CACHED_ARRAYS   = "$BRAIN_REPO\cached_data_arrays"
$TRANSCRIPTS_DIR = "$BRAINTREEBANK_DIR\transcripts"

# --- Task + global index tag ---
$TASK       = "word_onset"
$GLOBAL_IDX = "idx08_all_5_subjects_word_onset"

# ===============================
# 1) Trial selection JSON (all 5)
# ===============================

$null = New-Item -ItemType Directory -Force -Path "$REPO_DIR\trial_selections"

$brainRuns = @{
    "sub_1" = @("trial000", "trial001", "trial002");
    "sub_2" = @("trial000", "trial001", "trial002", "trial003");
    "sub_3" = @("trial000", "trial001", "trial002");
    "sub_4" = @("trial000", "trial001", "trial002");
    "sub_5" = @("trial000");
}

$BRAIN_RUNS_JSON = "$REPO_DIR\trial_selections\speech_allsubs_word_onset.json"
$brainRuns | ConvertTo-Json | Set-Content -Encoding utf8 $BRAIN_RUNS_JSON

Write-Host "Created brain_runs JSON at: $BRAIN_RUNS_JSON"
Get-Content $BRAIN_RUNS_JSON
Write-Host ""

# =============================================
# 2) Write multi-subject multi-channel data
#    (word_onset, all 5 subjects)
# =============================================

$ALL_OUT_DIR = "$REPO_DIR\saved_examples\all_${TASK}_$GLOBAL_IDX"

python -m data.write_multi_subject_multi_channel `
  "+data_prep=pretrain_multi_subj_multi_chan_template" `
  "++data_prep.task_name=$TASK" `
  "++data_prep.brain_runs=$BRAIN_RUNS_JSON" `
  "++data_prep.electrodes=$REPO_DIR/electrode_selections/clean_laplacian.json" `
  "++data_prep.output_directory=$ALL_OUT_DIR" `
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

Write-Host ""
Write-Host "Finished writing multi-subject word_onset data to: $ALL_OUT_DIR"
Write-Host ""

# =======================================
# 3) Per-subject manifests + PopT runs
# =======================================

# Define subject configs: id + a short index suffix
$subjects = @(
    @{ Id = "sub_1"; Suffix = "sub1_word_onset"; },
    @{ Id = "sub_2"; Suffix = "sub2_word_onset"; },
    @{ Id = "sub_3"; Suffix = "sub3_word_onset"; },
    @{ Id = "sub_4"; Suffix = "sub4_word_onset"; },
    @{ Id = "sub_5"; Suffix = "sub5_word_onset"; }
)

foreach ($s in $subjects) {

    $SUBJECT    = $s.Id
    $SUFFIX     = $s.Suffix
    $SUBJ_IDX   = "${GLOBAL_IDX}_${SUFFIX}"   # e.g., idx07_allsubs_word_onset_sub1_word_onset
    $SUBJ_OUT   = "$REPO_DIR\saved_examples\${SUBJECT}_${TASK}_cr_$SUBJ_IDX"

    Write-Host "==============================="
    Write-Host "Subject: $SUBJECT"
    Write-Host "Index:   $SUBJ_IDX"
    Write-Host "==============================="

    # 3.1) Subject-specific manifest
    python -m data.make_subject_specific_manifest `
      "+data_prep=subject_specific_manifest" `
      "++data_prep.data_path=$ALL_OUT_DIR" `
      "++data_prep.subj=$SUBJECT" `
      "++data_prep.out_path=$SUBJ_OUT"

    Write-Host "Created subject-specific data at: $SUBJ_OUT"
    Write-Host ""

    # 3.2) Run PopT downstream decoding

    $N    = 1
    $NAME = "popt_brainbert_stft_${TASK}_$SUBJ_IDX"

    # Results dir per subject
    $RESULTS_DIR = "$REPO_DIR\outputs\${SUBJECT}_${TASK}_${SUBJ_IDX}_popt_brainbert_stft_top${N}"

    python run_train.py `
      "+exp=multi_elec_feature_extract" `
      "++exp.runner.results_dir=$RESULTS_DIR" `
      "++exp.runner.save_checkpoints=True" `
      "++model.frozen_upstream=True" `
      "+task=pt_feature_extract_coords" `
      "+criterion=pt_feature_extract_coords_criterion" `
      "+preprocessor=empty_preprocessor" `
      "+data=pt_supervised_task_coords" `
      "++data.data_path=$SUBJ_OUT" `
      "++data.saved_data_split=$REPO_DIR/saved_data_splits/${SUBJECT}_${TASK}_fine_tuning_$SUBJ_IDX" `
      "++data.sub_sample_electrodes=$REPO_DIR/electrode_selections/clean_laplacian.json" `
      "+model=pt_downstream_model" `
      "++model.upstream_path=$PRETRAINED_POPT"

    Write-Host ""
    Write-Host "Finished training for $SUBJECT. Results in:"
    Write-Host "  $RESULTS_DIR"
    Write-Host ""
}

Write-Host "All subjects processed."
