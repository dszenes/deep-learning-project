library(keras)
library(cloudml)

FLAGS <- flags(
  flag_integer("n_neurons", 100)
)

generator <- image_data_generator(rescale = 1 / 255, validation_split = 0.2)

train_generator <- flow_images_from_directory(
  directory = gs_data_dir_local("gs://deep-unil-314206/data/data/train/"),
  generator = generator,
  target_size = c(300, 300),
  batch_size = 8,
  subset = "training"
)

valid_generator <- flow_images_from_directory(
  directory = gs_data_dir_local("gs://deep-unil-314206/data/data/train/"),
  generator = generator,
  target_size = c(300, 300),
  batch_size = 8,
  subset = "validation"
)

train_generator$num_classes

model1 <- keras_model_sequential() %>%
  layer_conv_2d(
    filter = 32,
    kernel_size = c(3, 3),
    input_shape = c(300, 300, 3),
    activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filter = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filter = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filter = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(FLAGS$n_neurons, activation = "relu") %>%
  layer_dense(2, activation = "softmax")

model1 %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_rmsprop(lr = 0.0001),
  metric = "accuracy"
)

model1 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = train_generator$n / train_generator$batch_size,
  epochs = 30,
  validation_data = valid_generator,
  validation_steps = valid_generator$n / valid_generator$batch_size,
  callbacks = callback_early_stopping(patience = 7,
                                      restore_best_weights = TRUE)
  
)

save_model_hdf5(model1, "model.hdf5")