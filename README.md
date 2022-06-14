# MAD Tools
A variety of tools for the S/D-MAD tasks.
<br>
All tools are written in Python language.

## Differential Morphing Attack Detection
- Reference paper: <i>Deep Face Representations for Differential Morphing Attack Detection (TIFS 2020)</i>  

### Image pre-processing
The D-MAD classifier is trained on embeddings extracted with the DNN presented in 

You can extract features through the ``extract_features.py`` script available in this repo. 
Follow inside comments for an easy use.<br>
Before, you have to install the following packages:
- ``Mxnet`` (tested version 1.4.0 running on CPU)
- ``OpenCV 4.4.0`` 

and other minor packages depending on your original setup (like ``tqdmm``, ...).

Click [here](https://miatbiolab.csr.unibo.it/wp-content/uploads/downloads/model-0000.params) to download the ArcFace parameters. 
For simplicity, put the file in the ``feature_extraction`` directory.

### Classifier
- SVM with rbf kernel
- Machine learning tool: ``scikit-learn 0.23.2``  
- Trained on PMDB dataset; data not balanced (280 genuine, 1108 impostor)
- Load the classifier file through the ``pickle`` package. 
Example:
```
import pickle
with open(<path>, 'wb') as f:
    classifier = pickle.load(f)
...
classifier.predict()
```
    
### Digital images

| Train dataset         | Train-Test Images          | Alpha         | Couples with  | EER on MorphDB | Model                                      |
| --------------------- | :------------------------: |:-------------:|:-------------:|:--------------:|:------------------------------------------:|
| PMDB                  | Digital-Digital            | 0.55          | Criminal      | 0.0%           |[link](Models/svm_rbf_digital_cri.pkl)      |
| PMDB                  | Digital-Digital            | 0.55          | Accomplice    | 0.0%           |[link](Models/svm_rbf_digital_acc.pkl)      |
| PMDB                  | Digital-Digital            | 0.55          | Both          | 0.0%           |[link](Models/svm_rbf_digital_bot.pkl)      |


### Printed and Scanned images

| Train dataset        | Train-Test Images          | Alpha         | Couples with  | EER on MorphDB | Model                                      |
| -------------------- | :------------------------: |:-------------:|:-------------:|:--------------:|:------------------------------------------:|
| PMDB                 | P&S-P&S                    | 0.55          | Criminal      | 0.0%           |[link](Models/svm_rbf_pes_cri.pkl)          |
| PMDB                 | P&S-P&S                    | 0.55          | Accomplice    | 0.0%           |[link](Models/svm_rbf_pes_acc.pkl)          |
| PMDB                 | P&S-P&S                    | 0.55          | Both          | 0.0%           |[link](Models/svm_rbf_pes_bot.pkl)          |