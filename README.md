# DeMakeup

[https://deeplearning4men.github.io/demakeup/](https://deeplearning4men.github.io/demakeup/)

Machine learning can perform many tasks at super-human level. [In one work](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4100743/), the researchers applied machine learning to recognise DISGUISED FACES and achieved human level performance. [A very recent work](https://www.wired.com/2016/09/machine-learning-can-identify-pixelated-faces-researchers-show/) suggests that AI can recognise human faces even when they are pixelated. Well then, why not apply the immense power of machine learning to remove cosmetics?

A dataset had been made for the machine to learn from. It was created by collecting faces without makeup then applying filters to virtually apply cosmetics on them (There are many apps for that purpose). After that, the job of the machine is to figure out how to do the reverse direction work. Currently, the dataset consists of 364 images. (This is a TINY dataset, I had to do all the work by myself!) In most of the images, smokey eye makeup style had been applied. Hence the current version of this DeMakeup is best trained to remove makeup in such style.

This work is powered by [Keras](https://keras.io/) and [Keras.js](https://transcranial.github.io/keras-js/).
