## Classification of extreme weather events Kaggle competition 1


### Getting Started

After you clone or download the code you should setup a virtual environment and
istall all dependencies present in the file requirements.txt

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
    sudo python3 -m pip sintall virtualenv
    ```
  * create an empty virtual environment
    ```sh
    python3 -m pip virtualenv myvenv
    ```
  * acvtivate your new virtual environment
    ```sh
    source /myvenv/bin.activate
    ```
  * install all the dependencies from requirement.txt
    ```sh
    pip install -r requirements.txt
    ```

## Usage

You can generate the predictions and evaluation on the validation set and generate the submission file for any model by calling any of the following functions defined in main.py at the end of the file 
- my_logistic_reg() : my implemented version of logistic regression present in logistic_reg.py
- gaussian_naive_bayes()
- scikit_logistic_reg()
- decision_tree()
- random_forest()
- gradient_boosting()
- dnn()
- svm()
- combine_predictions()

They are all commented at the end of the code so all you have to do is uncomment the selected method
and run the main
  ```sh
  python main.py
  ```