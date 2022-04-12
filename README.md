# RFSR_TGRS
The official code of hyperspectral image super-resolution method
```
@article{wang2021hyperspectral,
  title={Hyperspectral Image Super-Resolution via Recurrent Feedback Embedding and Spatial--Spectral Consistency Regularization},
  author={Wang, Xinya and Ma, Jiayi and Jiang, Junjun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--13},
  year={2021},
  publisher={IEEE}
}
```
## Testing
download the pretrained model an the test images from https://www.dropbox.com/sh/jqhk2i9lzi3hjna/AAAc9iDy-nxLh4wLbGGw3iXHa?dl=0
```
python main.py --phase='test' --test_model=model_path --test_dir=image_path
```

