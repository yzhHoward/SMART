# SMART
Official implementation of SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction

## Data Preparation
- Cardiology: https://physionet.org/content/challenge-2012/1.0.0/
- Sepsis: https://physionet.org/content/challenge-2019/1.0.0/
- MIMIC-III: https://physionet.org/content/mimiciii/1.4/

You need to follow the instructions on the PhysioNet website to access the data.

## Data Preprocessing

For Cardiology and Sepsis, please follow the jupyter notebook in the `data` folder to preprocess the data. Please move the files from zips in Cardiology to `raw` folder before running the notebook.

For MIMIC-III, please follow the instructions in the [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) repository to extract data from the MIMIC-III database. Notably, we adopt different settings when generating decompensation and length-of-stay data. Please replace the original `mimic3benchmark/scripts/create_decompensation.py` and `mimic3benchmark/scripts/create_length_of_stay.py` with the scripts in the `data/MIMIC-III` folder. After that, you can run the scripts in the jupyter notebook in the `data/MIMIC-III` folder to generate the tasks.

## Training and Evaluation

To train the model, please run the following command:

```bash
bash run.sh
```

## Citation
```
@inproceedings{yu2024smart,
  title={SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction},
  author={Yu, Zhihao and Chu, Xu and Jin, Yujie and Wang, Yasha and Zhao, Junfeng},
  booktitle={NeurIPS 2024},
  year={2024}
}
```
