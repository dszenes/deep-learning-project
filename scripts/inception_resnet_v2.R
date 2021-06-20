library(keras)
library(cloudml)

generator <- image_data_generator(rescale = 1 / 255, validation_split = 0.2)

train_flow <- flow_images_from_directory(
  directory = gs_data_dir_local("gs://deep-unil-314206/data/data/train/"),
  generator = generator,
  target_size = c(300, 300),
  batch_size = 8,
  subset = "training"
)

valid_flow <- flow_images_from_directory(
  directory = gs_data_dir_local("gs://deep-unil-314206/data/data/train/"),
  generator = generator,
  target_size = c(300, 300),
  batch_size = 8,
  subset = "validation"
)

model_base <- application_inception_resnet_v2(
  include_top = FALSE,
  weights = "imagenet", 
  input_shape = c(300, 300, 3)
)

freeze_weights(model_base)

inception_resnet_v2 <- keras_model_sequential() %>%
  model_base %>%
  layer_flatten() %>%
  layer_dense(
    units = 200,
    activation = "relu",
    kernel_regularizer = regularizer_l2(l = 0.01)
  ) %>%
  layer_dropout(0.3) %>% 
  layer_dense(
    units = 100,
    activation = "relu",
    kernel_regularizer = regularizer_l2(l = 0.01)
  ) %>%
  layer_dense(units = 2, activation = "softmax")

inception_resnet_v2 %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.00001), 
  loss = loss_categorical_crossentropy,
  metric = "accuracy"
)

inception_resnet_v2 %>% fit_generator(
  generator = train_flow,
  steps_per_epoch = train_flow$n / train_flow$batch_size,
  epoch = 30,
  validation_data = valid_flow,
  validation_steps = valid_flow$n / valid_flow$batch_size,
  callbacks = callback_early_stopping(patience = 7,
                                      restore_best_weights = TRUE)
)



save_model_hdf5(inception_resnet_v2, "model.hdf5")