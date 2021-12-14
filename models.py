import tensorflow as tf

SAMPLE_LIMIT = 160155


class LogSpectrogram(tf.compat.v1.keras.layers.Layer):

    def __init__(self, fft_size, hop_size=None, **kwargs):
        super(LogSpectrogram, self).__init__(**kwargs)
        self.fft_size = fft_size
        if hop_size is None:
            self.hop_size = fft_size//4
        else:
            self.hop_size = hop_size

    def build(self, input_shape):
        super(LogSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def amplitude_to_db(magnitude, amin=1e-16, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.amplitude_to_db.html
            """
            magnitude = tf.math.square(tf.abs(magnitude))
            ref_value = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        spectrograms = tf.signal.stft(tf.cast(waveforms, tf.float32),
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=False)

        log_spectrograms = amplitude_to_db(spectrograms)
        
        # Standardise Output
        log_spectrograms = (log_spectrograms - tf.keras.backend.mean(log_spectrograms))/tf.keras.backend.std(log_spectrograms)
        
        return log_spectrograms
    
    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size
        }
        config.update(super(LogSpectrogram, self).get_config())

        return config


def inverse_stft(n_fft, hop_size, tensor):
    return tf.signal.inverse_stft(tensor, n_fft, hop_size)


def build_generator(seed_size=5000, n_blocks=2, kernel_size=10, stride=5, d=64):
    # Define reuseable blocks
    def block(index):
        return [
            tf.compat.v1.keras.layers.Conv1DTranspose(
                d*(2**(n_blocks-index-1)), 
                kernel_size,
                strides=stride
            ),
            tf.compat.v1.keras.layers.ReLU(),
            tf.compat.v1.keras.layers.BatchNormalization()
        ]
    # Build model with specified number of blocks
    layers = [
        tf.compat.v1.keras.layers.Input([seed_size,]),
        tf.compat.v1.keras.layers.Dense(1280*(2**(n_blocks+1))*d),
        tf.compat.v1.keras.layers.Reshape([1280,(2**(n_blocks+1))*d])
    ]
    for idx in range(n_blocks):
        layers += block(idx) 
    layers += [
        tf.compat.v1.keras.layers.Conv1DTranspose(
            1, 
            kernel_size, 
            strides=stride
        )
        # tf.compat.v1.keras.layers.Activation('tanh')
    ]
    model = tf.keras.models.Sequential(layers, name='audio_generator')
    # Summarise model architecture 
    print(model.summary())
    return model


def build_discriminator(input_shape=[SAMPLE_LIMIT, 1, ], n_blocks=3, kernel_size=30, stride=5, d=64):
    # Define reuseable blocks
    def block(index):
        return [
            tf.compat.v1.keras.layers.Conv1D(
                d*(2**(index)),
                kernel_size,
                strides=stride
            ),
            tf.compat.v1.keras.layers.LeakyReLU(),
            tf.compat.v1.keras.layers.BatchNormalization()
        ]    
    # Build model with specified number of blocks
    layers = [
        tf.compat.v1.keras.layers.Input(input_shape)
    ] 
    for idx in range(n_blocks):
        layers += block(idx) 
    layers += [
        tf.compat.v1.keras.layers.Conv1D(
            128, 
            kernel_size, 
            strides=stride
        ),
        tf.compat.v1.keras.layers.GlobalAveragePooling1D(),
        tf.compat.v1.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.models.Sequential(layers, name='audio_discriminator')
    print(model.summary())
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return 0.5*total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def build_gan(generator, discriminator, optimiser):
    discriminator.trainable = False
    model = tf.keras.models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=optimiser, loss=cross_entropy)
    return model


def build_classifier(n_classes=20,
                     input_len=SAMPLE_LIMIT,
                     stem=None,
                     optimiser='adam',
                     loss='categorical_crossentropy',
                     conv_activation='relu'):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input((input_len, )))
    model.add(LogSpectrogram(2048))

    model.add(tf.keras.layers.Conv1D(32, 3))
    model.add(tf.keras.layers.Conv1D(32, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(32, 3))
    model.add(tf.keras.layers.Conv1D(32, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(64, 3))
    model.add(tf.keras.layers.Conv1D(64, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(64, 3))
    model.add(tf.keras.layers.Conv1D(64, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(128, 3))
    model.add(tf.keras.layers.Conv1D(128, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(128, 3))
    model.add(tf.keras.layers.Conv1D(128, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(256, 3))
    model.add(tf.keras.layers.Conv1D(256, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(256, 3))
    model.add(tf.keras.layers.Conv1D(256, 3, activation=conv_activation))
    model.add(tf.keras.layers.BatchNormalization())

    # model.add(tf.keras.layers.Conv1D(512, 3))
    # model.add(tf.keras.layers.Conv1D(512, 3, activation=conv_activation))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv1D(512, 3))
    # model.add(tf.keras.layers.Conv1D(512, 3, activation=conv_activation))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer=optimiser, loss=loss)
    print(model.summary())

    return model
