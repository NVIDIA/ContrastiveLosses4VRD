FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel
RUN apt-get update --fix-missing
RUN apt-get install -y software-properties-common
RUN apt-get install -y libsm6 libxext6 libxrender1 libfontconfig1
RUN pip install --upgrade pip
RUN pip install Cython matplotlib numpy scipy pyyaml packaging tensorboardX scikit-image pillow tqdm gensim
RUN pip install pycocotools
RUN conda install opencv