# dyscipher-OCR

A custom built OCR for dyslexic students

- Initial baby OCR model implemented

### Instructions to train the model:

- Download the dataset (http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)[here] - The EnglishHnd.tgz (13.0 MB) was used to spin up a model quickly
- Extract the files somewhere else, create a folder called `dataset` in this directory, and copy paste all the files inside tgz inside the direcftory.
- The directory should now look like
  ````|_____  dataset
                |______Sample001
                |______Sample002
                |__....Sample062
  |_____ remaining python files```
  ````
- run train_test_split.py (python3)
- run train.py if you want a model. (create a folder called models)
- run driver.py to perform OCR detection. (change path to image accordingly inside driver.py)
