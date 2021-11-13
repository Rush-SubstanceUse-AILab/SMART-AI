# Substance Misuse Algorithm for Referral to Treatment using Artificial Intelligence (SMART-AI)

Introduction 
- 

SMART-AI classifier identifies alcohol, opioid and non-opioid substance misuse from the Electronic Health Record of Hospitalized Patients. The first 24hr of clinical notes are needed as an input, which should be first processed using Apache cTAKES to concept map the raw with UMLS into Concept Unique Identifiers (CUIs).


Premilinary data processing steps:

cTAKES:

Download cTAKES from https://ctakes.apache.org/downloads.cgi cTAKES comes with default dictionary, this dictionary can also be cutomized creating own version. Our dictionary consists of rxnorms, snomedCT and drugbank but default dictionary also works well. Process the input data using cTAKES, this will create xmi file. CUIS from the xmi file can be extracted to create a .txt file, which will be input data to the model. 

Model:

Steps for model execution
1) Clone the repo
2) Open the config file inside the CODE folder to provide the data and model location
3) execution command inside the CODE folder -> python3 ml_prediction.py config.cfg
4) result will output in the same working directory


Libraries

1) python3
2) pickle
3) pytorch > 1.13
4) pandas
5) numpy

