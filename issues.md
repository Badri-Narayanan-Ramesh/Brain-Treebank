Nice, this is actually good progress — the pipeline is doing the right thing now and only dying on a memory issue:

numpy.core._exceptions._ArrayMemoryError: Unable to allocate 19.1 GiB for an array with shape (91, 2, 14065377) and data type float64

That’s coming from:

h5_data_reader.laplacian_rereference → filter_data → notch_filter → signal.lfilter(...)


So:
PopT is loading all 91 electrodes × 2 neighbors × 14M samples for the entire trial at 2,048 Hz, in float64, and then applying a notch filter. That single array is ~19 GB → your 32 GB machine chokes.

We don’t need fancy rereferencing to just get a working baseline on sub_1 / trial000 & 002, so the easiest fix is:

Disable laplacian rereferencing for this run → no giant filtered array → no 19 GB allocation.

We can do this purely via Hydra override, no file edits needed.

SOLUTION "++data.rereference=None"

That should:

skip laplacian reref,

avoid the 19 GB filter allocation,

finish writing features for trial000 and then trial002.

ISSUE 2 

The new error:

numpy.core._exceptions._ArrayMemoryError: Unable to allocate 51.1 GiB for an array with shape (7356, 91, 10240)


is happening here:

all_word_samples = np.stack([filtered_data[:, start:end] for (start,end) in word_intervals])


So for trial000 it’s trying to build:

7,356 intervals

× 91 electrodes

× 10,240 samples (≈ 5 seconds at 2048 Hz)

in float64 → ~51 GB just for this stack 😵

On a 32 GB laptop, that’s never going to fly with the full official config.

To still get a PopT word-onset baseline on your box, we’ll:

Turn off rereference ✅ (already did, rereference=None)

Shrink the window duration (5s → 1s)

Subsample the number of intervals (e.g., use ~10% of them)

This gives you a smaller but faithful version of the official pipeline: same code path, just fewer/shorter windows.

