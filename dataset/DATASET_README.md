# Data Collection Guide

This directory contains tools for collecting and validating sign language gesture data for the Motion-Capture-Sign-Language-Translation project.

## Directory Structure

```
dataset/
├── Halo/
├── Makan/
├── Minum/
├── Tolong/
└── TerimaKasih/
```

## Available Scripts

### 1. Data Collection (`collect_data.py`)

Capture gesture samples from webcam with real-time quality validation.

**Usage:**
```bash
# Collect for a specific class
python -m src.collect_data --class Halo --target 200

# Collect for all classes (interactive)
python -m src.collect_data --target 200

# Show current statistics
python -m src.collect_data --stats

# Disable display window (headless mode)
python -m src.collect_data --class Halo --no-display
```

**Controls:**
- `SPACE` - Capture current frame
- `Q` - Quit collection
- `R` - Reset sample count for current class

**Quality Validation:**
- Blur detection (Laplacian variance)
- Brightness check (min/max thresholds)
- Hand visibility verification
- Minimum hand area requirement

### 2. Dataset Validation (`validate_dataset.py`)

Validate dataset integrity and quality.

**Usage:**
```bash
# Full validation with quality check
python -m src.validate_dataset --check-quality

# Statistics only
python -m src.validate_dataset --stats-only

# Custom dataset path
python -m src.validate_dataset --dataset /path/to/dataset
```

### 3. Dataset Report (`dataset_report.py`)

Generate comprehensive dataset reports.

**Usage:**
```bash
# Generate both markdown and JSON reports
python -m src.dataset_report

# Custom collector name
python -m src.dataset_report --collector "Farhan Abdurrahman"

# Summary only
python -m src.dataset_report --summary-only
```

## Gesture Classes

| Class | Meaning | Description |
|-------|---------|-------------|
| Halo | Hello | Greeting gesture |
| Makan | Eat | Eating gesture |
| Minum | Drink | Drinking gesture |
| Tolong | Help | Request for assistance |
| TerimaKasih | Thank You | Gratitude gesture |

## Collection Requirements

### Image Quality
- Resolution: At least 640x480
- Blur score: Laplacian variance > 100
- Brightness: Between 30-220 (on 0-255 scale)
- Hand visibility: At least 70% of landmarks visible
- Minimum hand area: 5000 pixels

### Naming Convention
Format: `<ClassName>_<index>.jpg`

Examples:
- `Halo_001.jpg`
- `Halo_002.jpg`
- `Makan_001.jpg`

### Dataset Balance
Acceptable imbalance: ≤ 10% difference between largest and smallest class

### Minimum Requirements
- Per class: 100 samples (minimum)
- Per class: 200 samples (recommended)

## Recording Tips

1. **Lighting**: Capture in various lighting conditions
   - Natural light (window)
   - Artificial light (ceiling lamp)
   - Mixed lighting

2. **Background**: Use different backgrounds
   - Plain wall
   - Busy environment
   - Moving background (if possible)

3. **Hand Position**: Vary hand positions
   - Different distances from camera
   - Slight angle variations
   - Left and right hand (if applicable)

4. **Multiple Users**: Collect from multiple people if possible
   - Different hand sizes
   - Different skin tones
   - Different movement speeds

5. **Consistency**: Maintain recognizable gestures
   - Don't over-vary the gesture
   - Keep core movement pattern consistent
   - Small variations are OK

## Quick Start

1. Create dataset directory structure:
```bash
mkdir -p dataset/Halo dataset/Makan dataset/Minum dataset/Tolong dataset/TerimaKasih
```

2. Start collecting data:
```bash
python -m src.collect_data --target 200
```

3. Validate dataset:
```bash
python -m src.validate_dataset --stats-only
```

4. Generate report:
```bash
python -m src.dataset_report
```

## Output Files

- `dataset_report.json` - Machine-readable validation report
- `DATASET_REPORT.md` - Human-readable dataset documentation

## Notes

- Images are saved as JPEG for efficient storage
- MediaPipe is used for hand landmark detection
- Real-time feedback helps capture high-quality samples
- Batch collection mode cycles through all classes