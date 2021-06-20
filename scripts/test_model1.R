library(cloudml)
library(keras)


generator <- image_data_generator(rescale = 1 / 255)

test_generator = flow_images_from_directory(
  directory = gs_data_dir_local("gs://deep-unil-314206/data/data/test/"),
  generator = generator,
  target_size = c(300, 300),
  batch_size = 1,
  class_mode = 'categorical',
  shuffle = FALSE
)

model1 <-
  load_model_hdf5(
      here::here("runs/cloudml_2021_06_03_102739696-4/model.hdf5"),
    custom_objects = list(loss_categorical_crossentropy = loss_categorical_crossentropy)
  )

pred_model1 = predict_generator(
  object = model1,
  test_generator,
  verbose = 1,
  steps = test_generator$n
)

pred_model1 <- ifelse(pred_model1 >=0.5,1,0)

mat1 = matrix() 
for (i in 1:453) { 
  mat1[i] = 1
}
mat3 = matrix()
for (i in 1:262) { 
  mat3[i] = 0
}

test_set <- rbind(cbind(mat1), cbind(mat3))

