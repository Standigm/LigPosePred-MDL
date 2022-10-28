# Improving ligand pose prediction accuracy in noncognate receptors by combining physics-based and data-driven approaches

This is the code for "Improving ligand pose prediction accuracy in noncognate receptors by combining physics-based and data-driven approaches" paper [link](https://arxiv.org/).


# Dataset

Due to license issues, we cannot provide training datasets.
Testsets are available on [link]()

<br><br>

## Get Started

### 1. Set conda environment
```bash
conda env create -f requirements.yaml
# without openeye-toolkits
conda env create -f requirements_wo_openeye.yaml
```

### 2. Extract Pretrained Model Checkpoints
Extract checkpoint files.
```bash
tar -xvf checkpoints/ensemble3.tar.gz -C checkpoints
```

if openeye-toolkit is not installed, follow the comment [below](#openeye-license)


<br>

## Execute with example file.
Example files consist of a single pdb file, and a single sdf file with multiple ligands.

At the project root execute the following command
```bash
python pose_prediction.py --pdb_file examples/5AX9/5AX9_prep.pdb --sdf_file examples/5AX9/gold_60.sdf --output_prefix examples/5AX9/gold_60_score
```

You can set number of processes for fast data preparation
```bash
python pose_prediction.py --pdb_file examples/5AX9/5AX9_prep.pdb --sdf_file examples/5AX9/gold_60.sdf --output_prefix examples/5AX9/gold_60_score --num_processes 8
```

For GPU inference set --use_gpu


<br><br>

## Output File Format
Output file looks like the below. 

"ligand" column indicates the ligand title written in sdf file.
The results are sorted based on ligand title.

score_1,score_2 and score_3 are scores from indivisual model and score column shows averaged score (ensembled score)
```
ligand,score_1,score_2,score_3,score
AYQQUCOIONWNSZ-XZRNXFSWNA-M|AYQQUCOIONWNSZ-XZRNXFSWNA-M|sdf|1|dock1,0.950,0.873,0.903,0.909
AYQQUCOIONWNSZ-XZRNXFSWNA-M|AYQQUCOIONWNSZ-XZRNXFSWNA-M|sdf|1|dock2,0.925,0.854,0.838,0.872
AYQQUCOIONWNSZ-XZRNXFSWNA-M|AYQQUCOIONWNSZ-XZRNXFSWNA-M|sdf|1|dock3,0.778,0.895,0.793,0.822

...

```

<br><br>

## OpenEye License

An openeye license is required to use the code. If you don't have a license, you must obtain that license before using it. And set the OE_LICENSE environment variable.
```bash
export OE_LICENSE=<license_file_path>
```

You can skip the chemical feature extraction step for example file by extracting prep.tar.gz file.
```bash
tar -xvf examples/5AX9/prep.tar.gz -C examples/5AX9
```

And set --prep_odir to the directory where txt file exists.

```bash
python pose_prediction.py --pdb_file examples/5AX9/5AX9_prep.pdb --sdf_file examples/5AX9/gold_60.sdf --output_prefix examples/5AX9/gold_60_score --num_processes 8 --prep_odir examples/5AX9/prep
```

### 