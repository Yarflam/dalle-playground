FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# expose
EXPOSE 8080

# set working directory
WORKDIR /app

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install git
RUN apt-get install -y git

# update pip
RUN pip3 install --upgrade pip

# add requirements
COPY ./requirements.txt /app/requirements.txt

# install requirements
RUN pip3 install -r requirements.txt

# install jax[cuda]
# RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip3 list | grep "jax"
# RUN pip3 install --upgrade "jax[cuda]">=0.3
# RUN pip3 install --upgrade jaxlib>=0.3.10
# RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
# RUN pip install https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.3.10-cp38-none-manylinux2014_x86_64.whl
# RUN pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.10+cuda11.cudnn82-cp38-none-manylinux2014_x86_64.whl

# add source code
COPY . /app

# run server
CMD python3 app.py --port 3010 --model_version mini
