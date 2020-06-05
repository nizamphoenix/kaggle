This competition is the 3rd of its kind  organized by the Jigsaw team, a subsidiary of Google; 
the previous two hosted in 2018, the Toxic Comment Classification Challenge & in 2019, the Unintended Bias in Toxicity Classification respectively.  

In this competition, the objective is to use only english data to train model & run toxicity predictions on multiple languages, which can be done using multilingual models using TPUs.  

The probability of a comment being toxic is predicted across multiple languages. A toxic comment would receive a 1.0. A benign, non-toxic comment would receive a 0.0. In the test set, all comments are classified as either a 1.0 or a 0.0.

The xlm-roberta model from huggingface module is used to train the data; data from previous competitions is used along with other augmented data for the competitions.

