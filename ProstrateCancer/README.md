This [challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment) is about identifying cancerous tissues from benign ones and rate its severity(0 to 6), indicating severity of prostrate cancer. This challenge is conducted to push the frontiers of application of deep learning and image processing methods to histopathology.  

![image](./pandas.png)  
                                                 [image source](https://www.kaggle.com/c/prostate-cancer-grade-assessment)

### Approach:-  

The images provided are Whole Slide Images(WSI), high resolution images and thus cannot be directly processed by deep learning algorithms. Thus, a [technique](https://developer.ibm.com/technologies/data-science/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/) which has produced promising results over recent advancements in preprocessing the WSI images was used, which divides the entire image over a grid, generating tiles of the image, and then intelligently selects a few tiles based on how important each tile would be in determining the result. Efficiently sampling informative and representative patches is critical to achieving good performance. Moreover, all tiles are labelled with the label from the original WSI image, and then later fed to algorithms.  






- [x] 'facebookresearch/semi-supervised-ImageNet1K-models',model='resnext50_32x4d_ssl'--------0.45 kappa score  
- [x] 'zhanghang1989/ResNeSt', model='resnest50'                                      --------0.60 kappa score  
- [x] Se_Resnext_50_4-32                                                              --------


