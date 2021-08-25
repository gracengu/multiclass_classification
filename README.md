# Multi-class Classification Problem for Technical Assessment

This repo comprise of the codes developed for technical assessment. The goal of this assessment is to use the 
input variables to correctly classify or predict the target variable which is a multiclass categorical variable. 
There are a total of 150 input variables in the data. 

This repo is uploaded in github to demonstrate the use of agile and git throughout the exercise. 

## Table of Contents
* **1. About the Project**
* **2. Getting Started**
* **3. Set up your environment**
* **4. Open your Jupyter notebook**

## Structuring a repository
An integral part of having reusable code is having a sensible repository structure. That is, which files do we have and how do we organise them.
- Folder layout:
```bash
multiclass_classification
├── docs
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── conf.py
│       └── index.rst
├── src
│   └── analysis
│       └── __init__.py
│       └── processing.py
|       └── feature_engineer.py
|       └── statistical_analysis.py
|   └── train
│       └── __init__.py
│       └── train.py
|       └── Model.py
|   └── Config.py
├── .gitignore
├── README.md
├── environment.yml
├── setup.py
└── tox.ini
```

## 1. About the Project
The following is the summary of data scienc/ml/deep learning approaches used in this project:
  - <b><u>Placeholder</u></b>
  - <b><u>Placeholder</u></b>
  - <b><u>Placeholder</u></b>
  - <b><u>Placeholder</u></b>
  - <b><u>Placeholder</u></b>
  

## 2. Getting Started - Clone the repository locally

You may `git bash` at any preferred folder location and run the following command:

```bash
git clone https://github.com/gracengu/multiclass_classification.git
```
  
Alternatively, although not recommended, you can download the zip file of the repository at the top of the \
main page of the repository. If you prefer not to use git or don't have experience with it, this a good option.  

## 3. Set up your environment

Note: the following instructions are specifically for pip users only. 

The best practice is to create an isolated environment to avoid dependency conflicts in python. If this is the first \
time you're setting up your compute environment, please first install `pip` and `virtualenv`. 

```bash
python get-pip.py
pip install virtualenv
```

After installation, set up the virtual environment. If you have multiple python version installed in your PC, you may 
want to specify the python version you're using.

```bash
virtualenv --python=<path of python.exe with a specific version> python3.7_multiclass
```

To activate your environment, run the activate script from the `virtualenv` you have created: 

```bash
python3.7_multiclass\Scripts\activate
```

Change the path for tensorflow installation: 

```text
tensorflow-cpu @ file:///C:/Projects/2021/multiclass_classification/support/tensorflow_cpu-2.6.0-cp37-cp37m-win_amd64.whl
```

Please install all of the packages listed in the `requirement.txt` using the following command:

```bash
pip install -r requirement.txt
```

If there is error installing tensorflow, comment out tensorflow from requirements.txt and perform the following commands: 

```bash
pip install .\package\tensorflow_cpu-2.6.0-cp37-cp37m-win_amd64.whl
pip install -r requirements.txt
```


## 4. Open your Jupyter notebook

1. You will have to install a new IPython kernelspec to run the jupyter notebook in an isolated environment.
    
```bash
ipython kernel install --name python3.7_multiclass --user
```

You can change the `--name` to anything you want.

2. In the terminal, execute `jupyter notebook`.

Navigate to the notebooks directory and open notebook:
  - EDA: `Analysis.ipynb`
  - Modelling: `Train.ipynb`