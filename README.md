# Pdf2Rag2Rl

# Install
```
sudo apt-get update
sudo apt-get install tesseract-ocr
```

## AMD ROCm
```
pip install -r requirements-rocm.txt 
```

## CUDA and MPS
```
pip install -r requirements.txt 
```

# Running

Login & obtain access to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```
huggingface-cli login
```

If multiple GPUs:
```
export CUDA_VISIBLE_DEVICES=1
```

```
python src/main.py
```

# References

## YOLOv10
```
BibTeX
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```

## DocLayNet
```
@article{doclaynet2022,
  title = {DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis},  
  doi = {10.1145/3534678.353904},
  url = {https://arxiv.org/abs/2206.01062},
  author = {Pfitzmann, Birgit and Auer, Christoph and Dolfi, Michele and Nassar, Ahmed S and Staar, Peter W J},
  year = {2022}
}
```

## YOLOv10 - Document Layout Analysis
YOLOv10 trained on DocLayNet dataset
```
https://github.com/moured/YOLOv10-Document-Layout-Analysis
```