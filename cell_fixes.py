# =============================================================================
# FIXES FOR capstone_exploration.ipynb
# Copy each block into the corresponding cell.
# =============================================================================


# =============================================================================
# ► GPU SPEED SETUP  — paste this as a NEW cell right after cell a04 (imports)
#
#   SPEEDUP SUMMARY (single T4 GPU on Colab free tier):
#     - Mixed precision (float16): ~1.5–2× faster, same accuracy
#     - Dataset .cache():          ~1.2× faster after epoch 1 (avoids re-reading disk)
#     - Larger batch size:         ~1.3× faster (use 64 on GPU instead of 32)
#     - Combined on T4 GPU:        ~6–10× vs CPU  →  250 min → ~25–40 min
#
#   ON COLAB:  Runtime → Change runtime type → T4 GPU  (free)
#              Then just run normally — TF uses the GPU automatically.
#              No other code changes needed beyond this cell.
#
#   NOTE: Parallel training of all 5 models simultaneously is NOT practical
#   on a single GPU — TF allocates all GPU memory to the active model.
#   Sequential training is correct. The speedup below is per-model.
# =============================================================================

import tensorflow as tf

# ── Detect hardware ────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'GPU detected: {[g.name for g in gpus]}')
    # Allow memory growth so TF doesn't grab all VRAM at startup
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Mixed precision: stores weights as float32 but does compute in float16.
    # Free ~1.5–2× speedup on T4/V100/A100. Safe for this task.
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print('Mixed precision enabled: mixed_float16')

    # Larger batch fits in GPU memory — faster throughput
    BATCH_SIZE = 64   # override the 32 set in a06; delete this line on CPU
    print(f'Batch size set to {BATCH_SIZE} for GPU')
else:
    print('No GPU found — running on CPU. Consider Google Colab (Runtime → T4 GPU).')
    print('Estimated training time on CPU: 2–6 hours.')
    print('Estimated training time on Colab T4: ~25–40 minutes.')

print(f'TF version: {tf.__version__}')


# =============================================================================
# ► CELL a08  — also add .cache() to the dataset pipeline for GPU runs
#   (cache keeps the decoded images in RAM after the first epoch,
#    avoiding redundant disk reads. On Colab, RAM is ~12 GB — enough here.)
#
#   Change the _make_ds function inside cell a08 to this version:
# =============================================================================

def _make_ds(files, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(files), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()          # ← ADD THIS: cache decoded images in RAM
    if shuffle:
        ds = ds.shuffle(len(files), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# NOTE: cache() must come BEFORE batch() and AFTER map().
# The second shuffle (after cache) re-shuffles each epoch from the cached data.


# =============================================================================
# ► CELL a08  (currently titled "Load raw datasets")  ← THE ROOT CAUSE
#   WHAT CHANGED: Replaced image_dataset_from_directory + validation_split
#                 with a manual per-class stratified split using glob.
#
#   WHY IT WAS BROKEN: image_dataset_from_directory sorts all files
#   alphabetically across both classes — so the file list is:
#       [Benign/img001.jpg, ..., Benign/img6289.jpg,
#        Malignant/img001.jpg, ..., Malignant/img5590.jpg]
#   Taking the last 10% as validation = 1,187 files entirely from Malignant/.
#   Result: val set has ZERO Benign samples → AUC=0, precision=1.0,
#   recall=accuracy. The model cannot be evaluated properly.
#
#   THE FIX: glob each class directory separately, shuffle each, then take
#   10% from each → guaranteed stratified split.
# =============================================================================

import glob as _glob  # stdlib glob (not to be confused with any other)

# ── Collect files per class ────────────────────────────────────────────────
def _get_files(class_dir):
    files = []
    for pat in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        files.extend(_glob.glob(os.path.join(class_dir, pat)))
    return sorted(set(files))

benign_files    = _get_files(os.path.join(TRAIN_DIR, 'Benign'))
malignant_files = _get_files(os.path.join(TRAIN_DIR, 'Malignant'))

print(f'Total Benign    files found: {len(benign_files)}')
print(f'Total Malignant files found: {len(malignant_files)}')

# ── Stratified shuffle split ───────────────────────────────────────────────
rng = np.random.RandomState(SEED)

benign_arr    = np.array(benign_files);    rng.shuffle(benign_arr)
malignant_arr = np.array(malignant_files); rng.shuffle(malignant_arr)

n_bval = int(len(benign_arr)    * VAL_SPLIT)   # e.g. ~629
n_mval = int(len(malignant_arr) * VAL_SPLIT)   # e.g. ~559

benign_val,    benign_train    = benign_arr[:n_bval],    benign_arr[n_bval:]
malignant_val, malignant_train = malignant_arr[:n_mval], malignant_arr[n_mval:]

# ── Build file lists + labels ─────────────────────────────────────────────
train_files  = list(benign_train)      + list(malignant_train)
train_labels = [0.0]*len(benign_train) + [1.0]*len(malignant_train)

# Shuffle combined train list
combined = list(zip(train_files, train_labels))
rng.shuffle(combined)
train_files, train_labels = map(list, zip(*combined))

val_files  = list(benign_val)      + list(malignant_val)
val_labels = [0.0]*len(benign_val) + [1.0]*len(malignant_val)

# ── tf.data pipeline ──────────────────────────────────────────────────────
def _load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    label = tf.reshape(tf.cast(label, tf.float32), [1])
    return img, label

def _make_ds(files, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(files), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(_load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds_raw = _make_ds(train_files, train_labels, shuffle=True)
val_ds_raw   = _make_ds(val_files,   val_labels,   shuffle=False)

# Test set is already balanced — image_dataset_from_directory is fine here
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    interpolation='bilinear',
    shuffle=False,
)

# ── Class info ────────────────────────────────────────────────────────────
CLASS_NAMES = ['Benign', 'Malignant']

train_labels_arr    = np.array(train_labels)
N_BENIGN_TRAIN      = int((train_labels_arr == 0).sum())
N_MALIGNANT_TRAIN   = int((train_labels_arr == 1).sum())
N_TRAIN_TOTAL       = len(train_labels_arr)
CLASS_WEIGHT        = {
    0: N_TRAIN_TOTAL / (2 * N_BENIGN_TRAIN),
    1: N_TRAIN_TOTAL / (2 * N_MALIGNANT_TRAIN),
}

val_labels_arr  = np.array(val_labels)
N_BENIGN_VAL    = int((val_labels_arr == 0).sum())
N_MALIGNANT_VAL = int((val_labels_arr == 1).sum())

print(f'\nTrain : {N_BENIGN_TRAIN} Benign + {N_MALIGNANT_TRAIN} Malignant = {N_TRAIN_TOTAL} total')
print(f'Val   : {N_BENIGN_VAL} Benign + {N_MALIGNANT_VAL} Malignant = {len(val_labels)} total')
print(f'Class weights: {CLASS_WEIGHT}')
print(f'Train batches : {train_ds_raw.cardinality().numpy()}')
print(f'Val batches   : {val_ds_raw.cardinality().numpy()}')
print(f'Test batches  : {test_ds_raw.cardinality().numpy()}')


# =============================================================================
# ► CELL a25  (currently titled "Callbacks")
#   WHAT CHANGED: monitor='val_auc' → monitor='val_loss', mode='max' → 'min'
#   WHY: val_auc is broken in TF 2.19 Keras 3 — always 0, so EarlyStopping
#        never fires. val_loss is always reliable.
# =============================================================================

def get_callbacks(model_save_path=None, phase='phase1'):
    '''
    Returns Keras callbacks. Monitors val_loss (always reliable).
    Final AUC is computed via sklearn in evaluate_model() — not needed here.
    '''
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PAT,
            mode='min',
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if phase == 'phase2':
        cbs.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                mode='min',
                min_lr=1e-7,
                verbose=1,
            )
        )
    if model_save_path:
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=0,
            )
        )
    return cbs


print('get_callbacks() defined.')


# =============================================================================
# ► CELL a26  (currently titled "Phase 1 training")
#   WHAT CHANGED: METRICS = [...] global → def get_metrics(): return [...]
#                 Both compile() calls use get_metrics() not METRICS
#   WHY: Reusing the same stateful metric objects across compiles causes
#        internal state accumulation → garbage val_auc values.
# =============================================================================

def get_metrics():
    '''Returns FRESH metric instances. Never reuse across compile() calls.'''
    return [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


def phase1_train(model, train_ds, val_ds, loss, class_weight_dict=None,
                 save_path=None):
    '''Phase 1: Train classification head only (backbone frozen).'''
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=loss,
        metrics=get_metrics(),
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=get_callbacks(save_path, phase='phase1'),
        class_weight=class_weight_dict,
        verbose=1,
    )
    return model, history


def phase2_train(model, train_ds, val_ds, loss, class_weight_dict=None,
                 save_path=None):
    '''Phase 2: End-to-end fine-tuning (unfrozen backbone, lower LR).'''
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=PHASE2_LR,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=loss,
        metrics=get_metrics(),
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=get_callbacks(save_path, phase='phase2'),
        class_weight=class_weight_dict,
        verbose=1,
    )
    return model, history


print('get_metrics() defined.')
print('phase1_train() and phase2_train() defined.')


# =============================================================================
# AFTER PASTING ALL THREE FIXES:
# Delete any cached .pkl files before re-running the benchmark:
#
#   import shutil
#   shutil.rmtree('./capstone_results', ignore_errors=True)
#   import os; os.makedirs('./capstone_results', exist_ok=True)
# =============================================================================
