## CropHarvest crop or non-crop land Kaggle competition 1


### Getting Started

After you clone or download the code you should setup a virtual environment and all 
its dependencies present in the file requirements.txt

### Prerequisites

* python 3.9 (should also work with 3.8)
  * verify your python version
    ```sh
    python3 --version
    ```


* virtualenv
  * if you don't have pip installed
    ```sh
    sudo apt install python3-pip
    ```
  * install virtualenv
    ```sh
    sudo python3 -m pip intall virtualenv
    ```
  * create an empty virtual environment
    ```sh
    python3 -m pip virtualenv myvenv
    ```
  * acvtivate your new virtual environment
    ```sh
    source /myvenv/bin.activate
    ```
  * upgrade you virtual environment pip
    ```sh
    pip install --upgrade pip
    ```
  * install all the dependencies from requirement.txt
    ```sh
    pip install -r requirements.txt
    ```

## Setup

You need to download the 2 data files present in the 
Kaggle competition page and put them in the same repository as main.py 
- test_nolabels.csv
- train.csv


If they have a different name you should rename them to match these file names

## Usage

You can generate the predictions and evaluation on the validation set and generate the submission file for any model by calling any of the following functions defined in main.py at the end of the file 
- best_rf()
- best_xgboost()
- ada_boost()
- test_pca_xgboost()


They are all commented at the end of the code so all you have to do is uncomment the selected method
and run the main
  ```sh
  python main.py
  ```