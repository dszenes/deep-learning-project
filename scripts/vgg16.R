library(keras)
library(cloudml)

FLAGS <- flags(
  flag_integer("n_neurons", 100)
)

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

model_base <- application_vgg16(
  include_top = FALSE,
  weights = "imagenet", 
  input_shape = c(300, 300, 3)
)

freeze_weights(model_base)

vgg16_dropout <- keras_model_sequential() %>%
  model_base %>%
  layer_flatten() %>%
  layer_dense(
    units = FLAGS$n_neurons,
    activation = "relu",
    kernel_regularizer = regularizer_l2(l = 0.01)
  ) %>%
  layer_dropout(0.3) %>% 
layer_dense(
  units = FLAGS$n_neurons,
  activation = "relu",
  kernel_regularizer = regularizer_l2(l = 0.01)
) %>%
  layer_dense(units = 2, activation = "softmax")

vgg16_dropout %>% compile(
  optimizer = optimizer_rmsprop(0.00001), 
  loss = loss_categorical_crossentropy,
  metric = "accuracy"
)

vgg16_dropout %>% fit_generator(
  generator = train_flow,
  steps_per_epoch = train_flow$n / train_flow$batch_size,
  epoch = 30,
  validation_data = valid_flow,
  validation_steps = valid_flow$n / valid_flow$batch_size,
  callbacks = callback_early_stopping(patience = 7,
                                      restore_best_weights = TRUE)
)



save_model_hdf5(vgg16_dropout, "model.hdf5")

