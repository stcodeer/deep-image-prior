# Requirements

## gcc-10

```shell
$ sudo apt install gcc-10 g++-10 
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60 --slave /usr/bin/g++ g++ /usr/bin/g++-10
```

## pytorch_wavelets

```shell
$ git clone https://github.com/fbcotter/pytorch_wavelets
$ cd pytorch_wavelets
$ pip install .
```

## re2c

## ninja

## torch-dwconv



# Denoising Experiment

```shell
$ python test_denoising.py
```

