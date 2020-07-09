This [challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment) is about identifying cancerous tissues from benign ones and rate its severity(0 to 6), indicating severity of prostrate cancer. This challenge is conducted to push the frontiers of application of deep learning and image processing methods to histopathology.  

### Approach:-  

The images provided are Whole Slide Images(WSI), high resolution images and thus cannot be directly processed by deep learning algorithms. Thus, a [technique](https://developer.ibm.com/technologies/data-science/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/) which has produced promising results over recent advancements in preprocessing the WSI images was used, which divides the entire image over a grid, generating tiles of the image, and then intelligently selects a few tiles based on how important each tile would be in determining the result. Moreover, all tiles are labelled with the label from the original WSI image, and then later fed to algorithms.  



