File Descriptions :
===================
predict.py : Takes the path to a locaiton and predicts the output for files in that folder.
model.py   : Defines the model class.
dataset.py : Defines the test dataset class for feeding it into the model.
utils.py   : Pre-processing functions.
lib.py     : Contains imports.


Usage :
========
Specify the location by the '-t' Flag.


Sample Usage : 
===============
python predict.py -t 'test1'
python predict.py --test-location 'test2'

Output :
========
Output file is stored as 'output.txt' in the same folder as this Readme file.