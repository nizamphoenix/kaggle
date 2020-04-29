The channels are classified broadly into 2 categories depending on whether they have a low probability of opening(low conductance) or a high probability of opening(high conductance):-  

1. Batch1 and Batch2 represent low-probability channels: binary classification  
  -- the data instances recorded in batch1 and batch2 correspond to either one channel being open or closed, hence the name          low-probability channels.  
  -- Model-1 is dedicated to modelling data from batch1 and batch2 only.  
  
2.Other Batches represnet high-probabaility channels: multi-class classification  
  -- the data instances recorded in batch3 upto batch10 correspond to either multiple channels, upto 10 being open or none          being open i.e. closed, hence the name high-probability channels.  
  -- Model-2 is dedicated to modelling data from batch3 until batch10.  
  

### Model1: 
  DecisionTree classifying a channel as either open or closed.
### Model2:  
  1. XGBoost: gives ~46% accuracy.  
  2. Feed-forward neural network with 2 hidden layers: gives 51% accuracy.  
  3. 1D-CNN: 81% accuracy  
  4. 2 models, one for batch-5 and batch-10 and the other for batches 3,4,6,7,8,9  
