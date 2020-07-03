# SatelliteImageProcessing

In this project, i have demonstrated an accurate, inexpensive and scalable method to estimate the socio economic status of a geographical region from high-resolution satellite images using CNN, this method could ease efforts to track economic condition in developing countries. 

# UNet Model for Segmentation
Unet is a Fully Convolutional Neural Network Architecture which is used as state-of-art for Segmentation. Unet was ﬁrst developed for Biomedical image Segmentation by Olaf Ronneberger et al. Unet consist of two parts a encoder, which is used to capture the context of the image and a decoder, which is used to precisely localized the image. The encoder consist of traditional convolutions layers followed by max pooling layes and encoder consist of transpose convolutions. In encoder the model learns ’what’ information is in the image but lost the 'where' information which is recovered by the decoder part of the Unet. As it is Fully Convolution Neural Network it do not consist of any Dense layer hence, can take image of any size. 

In this project i have implemented a Unet Model to perform segmentation and obtain the features essential to evaluate the socio economic status of an geographical region such as avaliability water resources, vegetation, infrastructure development and transportation means like road & railways. A dropout of 40 percent is introduced to decrease the high variance and also a batch normalization before each upsampling and downsampling layer to tackle the variance and mean of the feature maps. For traning the model we have chosen Adam optimizer with 0.0001 learning rate and categorical cross entropy as loss function, with a batch size of 32 for around 50 epochs. The model is train on Google Colabs notebook which is a free cloud service and provide GPU service



