# TSID-Net: A Two-stage Single Image Dehazing Framework with Style Transfer and Contrastive Knowledge Transfer

Authors: Shilong Wang, Qianwen Hou, Jiaang Li, Jianlei Liu

## Prerequisites
Python 3.6 or above.

For packages, see requirements.txt.

### Getting started

- Install PyTorch 1.8 or above and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.
  
### TSID-Net Training and Test

- A one image train/test example is provided.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the TSID-Net model:
```bash
python Stage1_train_RS.py
python Stage2_1_train_Teachers.py
python Stage2_2_train_TS.py 
```
- Test the TSID-Net model:
```bash
python test_TS.py
```

