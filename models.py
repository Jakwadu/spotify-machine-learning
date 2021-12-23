import tensorflow as tf

SAMPLE_LIMIT = 160155

def inverse_stft(n_fft, hop_size, tensor):
    return tf.signal.inverse_stft(tensor, n_fft, hop_size)


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


def build_classifier(n_classes=20,
                     input_len=SAMPLE_LIMIT,
                     stem='conv',
                     optimiser=tf.keras.optimizers.Adam(learning_rate=3e-4),
                     loss='categorical_crossentropy',
                     conv_activation=tf.keras.layers.ReLU(),
                     conv_padding='causal'):

    assert stem == 'conv' or stem == 'lstm'

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input((input_len, )))
    model.add(LogSpectrogram(2048))

    if stem == 'conv':
        model.add(tf.keras.layers.Conv1D(32, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(32, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(32, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(32, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(64, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(64, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(64, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(128, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(128, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(128, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(128, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(256, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(256, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(256, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(256, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(512, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(512, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(512, 3, padding=conv_padding))
        model.add(tf.keras.layers.Conv1D(512, 3, padding=conv_padding))
        model.add(conv_activation)
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.GlobalAveragePooling1D())

    elif stem == 'lstm':
        model.add(tf.keras.layers.LSTM(1024))

    model.add(tf.keras.layers.Dropout(0.))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer=optimiser, loss=loss)
    print(model.summary())

    return model
