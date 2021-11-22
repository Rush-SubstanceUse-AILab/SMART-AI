# Substance Misuse Algorithm for Referral to Treatment using Artificial Intelligence (SMART-AI)

Introduction 
- 

SMART-AI classifier identifies alcohol, opioid, and non-opioid substance misuse from the Electronic Health Record of Hospitalized Patients. The first 24hr of clinical notes are input, which should be processed using Apache cTAKES to concept map the raw with UMLS into Concept Unique Identifiers (CUIs).

Preliminary data processing steps:
cTAKES:
Download cTAKES from https://ctakes.apache.org/downloads.cgi cTAKES comes with a default dictionary. Our dictionary consists of rxnorms and snomedCT, but the default dictionary also works well. Process the input data, first 24hr of clinical notes, using cTAKES to create xmi files. The xmi files will contain concept unique identifiers (CUIS) extracted to create a .txt file. Finally, the .txt file will be input into the model.

Model:
The model can run in a batch of patients data or just a single patient file. Keep the data inside an arbitrary folder with cuis for each patient in a .txt file in both cases. Provide the directory to this dataset in the config file.

Steps for model execution
Clone the repo or download it from the repo directory
Open the config file inside the CODE folder, which contains data and model directory location
Modify the directory location to the path where the actual data and model resides
From the CODE folder, execute the command: python3 ml_prediction.py config.cfg
Results will output in the same working directory, with a list of the file names and their prediction probabilities, prediction outcome(binary 0 for no misue and 1 for misuse) for each misuse status. The cut point is kept at 0.05 but can be changed easily from the config file.

Libraries:

python3
pickle
PyTorch > 1.13
pandas
NumPy
