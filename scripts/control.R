library(cloudml)
library(dplyr)

gs_copy(
  source = "/Users/.../data/",  #path
  destination = "gs://.../data/",
  recursive = TRUE
) #from local to cloud


######TWO BEST MODELS
#own architecture -> hyperparameter tuning only on the number of neurons,
cloudml_train(file = "scripts/model1.R", master_type = "standard_gpu", config = "scripts/tuning.yml")
job_collect("cloudml_2021_06_03_102739696", trials = "all")

#dropoutvgg16 with para
cloudml_train(file = "scripts/vgg16.R", master_type = "standard_gpu", config = "scripts/tuning.yml")
job_collect("cloudml_2021_06_06_145113483", trials = "all")

##OTHERSMODELS TESTED -> without hyperpara
cloudml_train(file = "scripts/raw vgg16.R", master_type = "standard_gpu")
job_collect("cloudml_2021_06_03_085630110")

cloudml_train(file = "scripts/inception_resnet_v2.R", master_type = "standard_gpu")
job_collect("cloudml_2021_06_08_071646012") #poor job -> 43% val_acc


####Louis' yaml error
reticulate::use_python("/usr/bin/python", required = T)
#or
reticulate::use_python(python = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7', required = T)

#check python verion
reticulate::py_config()

#if not working because numpy not find

# put in the console:

# check the python version : has to be 3.7 ->     python3 -V

#if numpy error
# instal numpy ->                                 pip3 install numpy
#                                                 pip3 show numpy



