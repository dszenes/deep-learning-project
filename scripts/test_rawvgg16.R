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

raw_vgg16 <-
  load_model_hdf5(
    here::here("runs/cloudml_2021_06_03_085630110/model.hdf5"),
    custom_objects = list(loss_categorical_crossentropy = loss_categorical_crossentropy)
  )

pred_raw_vgg16 = predict_generator(
  object = raw_vgg16,
  test_generator,
  verbose = 1,
  steps = test_generator$n
)

pred_raw_vgg16 <- ifelse(pred_raw_vgg16 >=0.5,1,0)