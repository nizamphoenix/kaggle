This [competition](https://www.kaggle.com/c/trends-assessment-prediction/submissions) was aimed at predicting age & four other assessment scores that helps to capture useful patient-level characteristics in clinical settings, for individualized treatments.    

The following data was provided to facilitate useful analytics,  
- functional: 3D functional spatial maps from resting-state functional-MRI(fMRI)     
- connectivity: static functional connectivity  network connectivity (FNC) matrices                 
- structural: source based morphometry (SBM) loading values from structural-MRI(sMRI)

sMRI is a technique for examining the anatomy and pathology of the brain, while [fMRI](https://www.ed.ac.uk/clinical-sciences/edinburgh-imaging/research/themes-and-topics/medical-physics/imaging-techniques/functional-mri) is used to examine brain activity. 
[functional connectivity](https://www.sciencedirect.com/topics/medicine-and-dentistry/functional-connectivity) is only the presence of statistical dependencies between two sets of neurophysiological data and does not incorporate any knowledge or assumptions about the structure and mechanisms of the neural system of interest.

#### Competition metric: 

Submissions are scored using  feature-weighted-normalized-absolute errors given by the formula below,  

<img src="https://render.githubusercontent.com/render/math?math=\text{score} = \sum_{j} w_j \left( \frac{\sum_i \text{abs}( y_{j_i} - \hat{y}_{j_i})}{\sum_i \hat{y}_{j_i}} \right)">   

[Here](https://github.com/nizamphoenix/kaggle/blob/master/TRends/score.py) is the code for the formula  

j--->age/domain1_var1/domain1_var2/domain2_var1/domain2_var2  (target variables)  
i--->data instance  
<img src="https://render.githubusercontent.com/render/math?math=y_{j_i}"> is the 'i'th data observation of 'j'th feauture.    
weights : [.3, .175, .175, .175, .175]  

#### Approach:  

I utilised only the tabular data and did not utilise the 3d-images provided. With just the tabular data, which is high-dimensional, it was treated as a predective modelling problem. Also, the train and test data were not iid(independent data points); however,since it is a prerequisite to ensure independence amongst the data points *adversarial validation* was performed to figure out the features that were causing trouble.  
This interdependence was figured out by building a robust classifier with training data tagged as 0 and the test data being tagged as 1, and calculating the AUC score.*Higher AUC score indicates more interdependence*, but lower AUC score is desired. Transforming the features or dropping them are a few ways to reduce AUC.
Thereafter with further sophisticated analysis, about 5 features were dropped which would reduce the dependebility among data points while giving good results.    
The following libraries/frameworks were used to produce predictive models,  

- [RapidsAI](https://github.com/nizamphoenix/kaggle/blob/master/TRends/train.py)  
Since data is high dimensional(1405 features), computations demanded more power, hence Rapids AI library by Nvidia was used to build regression models
with custom loss and metrics as provided by the competition hosts.  Each one of the 5 targets were modelled separately with Support vector regression, Elastic net and Random forest regressor and the final predictions were blended to produce the final result.   
There was about 0.1 of a difference between CV score and leaderboard score.  

- [lightGBM](https://github.com/nizamphoenix/kaggle/blob/master/TRends/lightgbm_CV.py)    
Any predictive modelling task would be incomplete without inclusion of gradient boosting, the lightGBM library is compatible with GPU and facilitates parallel training too. The most crucial element while using lightGBM is the set of hyperparameters which need to be tuned, for this *optuna*'s integration was used to select the best set of hyperparameters with the objective function and the loss fixed at *hinge* and *L1*, respectively.  

- [TabNet](https://github.com/nizamphoenix/kaggle/blob/master/TRends/tabnet_cv_train.py)  
It is a novel neural-network architechture propounded by Google that uses attention mechanism to perform better with tabular data at predictive modelling tasks. The version provided by *fastAi* was used  and it outperformed the above two approaches. However, unlike the previous two approached all 5 targets were jointly modelled --*multi-task learning*-- to obtain final results with 7 fold cross validation training. Also, tabnet was used to model individual targets, similar to the previous approaches, which resulted in poor results.  
There was about 0.0001 of a difference between CV score and leaderboard score.  

Moreover, the competition [metric](https://github.com/nizamphoenix/kaggle/blob/master/TRends/score.py) and [loss functions](https://github.com/nizamphoenix/kaggle/blob/master/TRends/losses.py) were customised.

