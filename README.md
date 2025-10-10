# Data-driven Precipitation Nowcasting Using Satellite Imagery (AAAI2025 ORAL)
### [Project Page](-) | [Dataset](https://drive.google.com/drive/folders/1VkOtALaj5cqTFkNy5NvXKE4z83elijOh?usp=drive_link) | [Paper](https://arxiv.org/abs/2412.11480) | 

**Official Note**:

- 2025.10.10
We moved to a new workplace, and unfortunately all the data was lost because my Google account from the previous institution was deleted, which also prevented me from sharing it.

Therefore, we re-refined and re-uploaded the dataset.
All of our code, datasets, and documentation have now been migrated to:
👉 https://huggingface.co/datasets/minseok96/SAR2RDR-Dataset/tree/main

You can find all preprocessing and postprocessing scripts there, along with detailed explanations.

---
<p align="center">
<img width="691" alt="스크린샷 2024-12-19 오후 4 13 17" src="https://github.com/user-attachments/assets/e4b4d125-2086-4ef8-a92e-ef5f900e13d7" />

</p>

---
## GK2A Preprocessing
```
preprocessing.py
```

## HSP Preprocessing
```
HSP_processing.py
```


### Training
```
pytohn3 main.py
```


### Citation

```
@article{park2024data,
  title={Data-driven Precipitation Nowcasting Using Satellite Imagery},
  author={Park, Young-Jae and Kim, Doyi and Seo, Minseok and Jeon, Hae-Gon and Choi, Yeji},
  journal={arXiv preprint arXiv:2412.11480},
  year={2024}
}```
