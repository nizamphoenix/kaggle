This competition is aimed at predicting age & four other assessment scores that will help to capture useful patient-level characteristics in clinical settings, for individualized treatments.    

The following data is provided to facilitate useful analytics,  
- 3D functional spatial maps from resting-state functional-MRI(fMRI)    ----->functional  
- static functional connectivity  network connectivity (FNC) matrices                 ----->connectivity   
- source based morphometry (SBM) loading values from structural-MRI(sMRI)---->structural  

sMRI is a technique for examining the anatomy and pathology of the brain, while [fMRI](https://www.ed.ac.uk/clinical-sciences/edinburgh-imaging/research/themes-and-topics/medical-physics/imaging-techniques/functional-mri) is used to examine brain activity. 
[functional connectivity](https://www.sciencedirect.com/topics/medicine-and-dentistry/functional-connectivity) is only the presence of statistical dependencies between two sets of neurophysiological data and does not incorporate any knowledge or assumptions about the structure and mechanisms of the neural system of interest.

#### Competition score: 

Submissions are scored using  feature-weighted-normalized-absolute errors given by the formula below,  

<img src="https://render.githubusercontent.com/render/math?math=\text{score} = \sum_{j} w_j \left( \frac{\sum_i \text{abs}( y_{j_i} - \hat{y}_{j_i})}{\sum_i \hat{y}_{j_i}} \right)">   

[Here](https://github.com/nizamphoenix/kaggle/blob/master/TRends/score.py) is the code for the formula  

j--->age/domain1_var1/domain1_var2/domain2_var1/domain2_var2  (target variables)  
i--->data instance  
<img src="https://render.githubusercontent.com/render/math?math=y_{j_i}"> is the 'i'th data observation of 'j'th feauture.    
weights : [.3, .175, .175, .175, .175]  
