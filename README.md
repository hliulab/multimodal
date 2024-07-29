## Multi-modal deep learning for predicting PD-L1 biomarker and immunotherapy response of esophageal cancer


### H&E slide preprocessing
We follow our former WSI preprocessing solution (https://github.com/hliulab/wsi2recurrence)

```python
# WSI Segmentation and Patching
python create_patches_fp.py --source ../slides/TCGA-BRCA
--patch_size 256
--save_dir ../tile_results
--patch
--seg
--tcga_flag

# tiling
python save_tiles.py --patch_size 256 --sample_number 100 --save_dir ../tiles_result

# Training contrastive learning model
python3 train_adco.py --dist_url=tcp://localhost:10001 --data ../tiles_result/tiles_20x
--save_path ../MODELS_SAVE
--model_path ../MODELS_SAVE
```

### Extracting patch-level features

```python
python extra_feat.py --file_path ../tile_results
--save_path ../feat
--model_path ../MODELS_SAVE/adco_tcga.pth.tar
```

### Radiomic Features Extraction

This project involves the extraction of radiomics features using the pyradiomics 3.1.0. The dataset includes features extracted from pre-treatment and early on-treatment CT scans.

1. **Radiomic Features:**
   - Radiomic features before treatment are stored in `dataset/Immunotherapy_before_feature.xlsx`.
   - Radiomic features of early on-treatment are stored in `dataset/ImmunotherapyCT_after_feature.xlsx`.

2. **Clinical and Radiomics feature Integration:**
   - The combined dataset, which includes clinical variables and radiomics features, is stored in `combined.xlsx`.

### Training for PD-L1 level prediction using multimodal features

```python
python train.py --task PD-L1 --files_path files
--coords_path coord.pt
--cli_and_rad_path features.xlsx
--label_path labels.xlsx
--save output
```

### Training for immunotherapy response prediction using multimodal features

```python
python train_immunotherapy.py --task Immunotherapy --files_path feat_Immunotherapy
--coords_path yh.pth
--cli_and_rad_path dataset/feature.xlsx
--combined_path dataset/combined.xlsx
--radiomics_before_path dataset/Immunotherapy_before_feature.xlsx
--radiomics_after_path dataset/Immunotherapy_after_feature.xlsx
--label_path Immunotherapy.xlsx
--save save
--model_path save/PD-L1_0.pth
```
