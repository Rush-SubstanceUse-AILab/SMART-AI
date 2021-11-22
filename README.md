# Substance Misuse Algorithm for Referral to Treatment using Artificial Intelligence (SMART-AI)

Introduction 
- 

SMART-AI classifier identifies alcohol, opioid and non-opioid substance misuse from the Electronic Health Record of Hospitalized Patients. The first 24hr of clinical notes are needed as an input, which should be first processed using Apache cTAKES to concept map the raw with UMLS into Concept Unique Identifiers (CUIs).


Premilinary data processing steps:

cTAKES:

Download cTAKES from https://ctakes.apache.org/downloads.cgi cTAKES comes with default dictionary, this dictionary can also be cutomized creating own version. Our dictionary consists of rxnorms and snomedCT, but default dictionary also works well. Process the input data, first 24hr of clinical notes using cTAKES, this will create xmi files. The xmi files will contain CUIS which can be extracted to create a .txt file. The .txt file will be input data to the model. 

Model:

The model can run in a batch of patient data or just a single patient file. In both the cases, the data should kept inside a folder with cuis for each patient in a .txt file. This directory to this folder must be given in the config file. 

Steps for model execution
1) Clone the repo or download from the repo directory
2) Open the config file inside the CODE folder, which contains data and model directory location
3) Modify the directory location to the path where the actual data and model resides
3) From the CODE folder execution the command: python3 ml_prediction.py config.cfg (this step scan also be done by loading the ml_prediction.py script in an IDE)
4) Results will output in the same working directory, with a list of file name and their prediction probability, prediction outcome(binary 0 for no misue and 1 for misuse) for each misuse status. The cut point can also be changed from the config file, default cutpoint is 0.05. 


Libraries

1) python3
2) pickle
3) pytorch > 1.13
4) pandas
5) numpy

