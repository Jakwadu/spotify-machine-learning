import tensorflow as tf

SAMPLE_LIMIT = 160155

def inverse_stft(n_fft, hop_size, tensor):
    return tf.signal.inverse_stft(tensor, n_fft, hop_size)


class LogSpectrogram(tf.keras.layers.Layer):

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


def conv_block(n_filters, filter_size, activation, dilation, padding):
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv1D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(tf.keras.layers.Conv1D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(activation)
    block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.Conv1D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(tf.keras.layers.Conv1D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(activation)
    block.add(tf.keras.layers.BatchNormalization())
    return block


def conv2d_block(n_filters, filter_size, activation, dilation, padding):
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(tf.keras.layers.Conv2D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(activation)
    block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.Conv2D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(tf.keras.layers.Conv2D(n_filters, filter_size, dilation_rate=dilation, padding=padding))
    block.add(activation)
    block.add(tf.keras.layers.BatchNormalization())
    return block


def wavenet_block(inputs, dilation_rate):
    conv = tf.keras.layers.Conv1D(32, 3, dilation_rate=dilation_rate, padding='same')(inputs)
    tanh = tf.keras.layers.Activation('tanh')(conv)
    sigmoid = tf.keras.layers.Activation('sigmoid')(conv)
    gated = tf.keras.layers.Multiply()([tanh, sigmoid])
    return gated


def build_classifier(n_classes=20,
                     input_len=SAMPLE_LIMIT,
                     stem='conv',
                     skip_spectrogram=False,
                     optimiser=tf.keras.optimizers.Adam(learning_rate=3e-4),
                     loss='categorical_crossentropy',
                     conv_activation=tf.keras.layers.Activation('relu'),
                     conv_padding='causal',
                     filter_size=1,
                     dropout_rate=0.):

    assert stem in ['conv', 'conv2d', 'wavenet', 'lstm']

    if stem == 'conv2d':
        conv_padding = 'same'

    inputs = tf.keras.layers.Input((input_len, ))

    if skip_spectrogram and stem != 'conv2d':
        processed_inputs = tf.keras.layers.Dense(512, activation=None)(inputs)
        processed_inputs = tf.keras.layers.Reshape((512, 1))(processed_inputs)
    else:
        processed_inputs = LogSpectrogram(2048)(inputs)

    if stem == 'wavenet':
        conv0 = conv_block(32, filter_size, conv_activation, 1, conv_padding)(processed_inputs)

        conv1 = wavenet_block(conv0, 2)
        conv1 = tf.keras.layers.Add()([conv0, conv1])

        conv2 = wavenet_block(conv1, 4)
        conv2 = tf.keras.layers.Add()([conv1, conv2])

        conv3 = wavenet_block(conv2, 6)
        conv3 = tf.keras.layers.Add()([conv2, conv3])

        add = tf.keras.layers.Add()([conv1, conv2, conv3])
        nonlinearity = tf.keras.layers.Activation('relu')(add)
        pool = tf.keras.layers.GlobalMaxPool1D()(nonlinearity)
        dropout = tf.keras.layers.Dropout(dropout_rate)(pool)

    elif stem == 'conv':
        conv0 = conv_block(32, filter_size, conv_activation, 1, conv_padding)(processed_inputs)
        conv1 = conv_block(32, filter_size, conv_activation, 1, conv_padding)(conv0)
        conv1 = tf.keras.layers.Add()([conv0, conv1])

        conv2 = conv_block(64, filter_size, conv_activation, 1, conv_padding)(conv1)
        conv3 = conv_block(64, filter_size, conv_activation, 1, conv_padding)(conv2)
        conv3 = tf.keras.layers.Add()([conv2, conv3])

        conv4 = conv_block(128, filter_size, conv_activation, 1, conv_padding)(conv3)
        conv5 = conv_block(128, filter_size, conv_activation, 1, conv_padding)(conv4)
        conv5 = tf.keras.layers.Add()([conv4, conv5])

        conv6 = conv_block(256, filter_size, conv_activation, 1, conv_padding)(conv5)
        conv7 = conv_block(256, filter_size, conv_activation, 1, conv_padding)(conv6)
        conv7 = tf.keras.layers.Add()([conv6, conv7])

        pool = tf.keras.layers.GlobalAveragePooling1D()(conv5)
        dropout = tf.keras.layers.Dropout(dropout_rate)(pool)

    elif stem == 'conv2d':
        conv2d_in = tf.keras.layers.Reshape((126, 1025, 1))(processed_inputs)
        conv2d_in = tf.keras.layers.experimental.preprocessing.Resizing(126, 126)(conv2d_in)

        conv0 = conv2d_block(16, filter_size, conv_activation, 1, conv_padding)(conv2d_in)
        conv1 = conv2d_block(16, filter_size, conv_activation, 1, conv_padding)(conv0)
        conv1 = tf.keras.layers.Add()([conv0, conv1])

        conv2 = conv2d_block(32, filter_size, conv_activation, 1, conv_padding)(conv1)
        conv3 = conv2d_block(32, filter_size, conv_activation, 1, conv_padding)(conv2)
        conv3 = tf.keras.layers.Add()([conv2, conv3])

        conv4 = conv2d_block(32, filter_size, conv_activation, 1, conv_padding)(conv3)
        conv5= conv2d_block(32, filter_size, conv_activation, 1, conv_padding)(conv4)
        conv5 = tf.keras.layers.Add()([conv4, conv5])

        pool = tf.keras.layers.GlobalAveragePooling2D()(conv5)
        dropout = tf.keras.layers.Dropout(dropout_rate)(pool)

    elif stem == 'lstm':
        # lstm1 = tf.keras.layers.LSTM(1024)(processed_inputs)
        # lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=1025)(processed_inputs, processed_inputs)
        pool = tf.keras.layers.GlobalMaxPool1D()(attention)
        dropout = tf.keras.layers.Dropout(dropout_rate)(pool)

    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(dropout)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=optimiser, loss=loss)
    print(model.summary())

    return model

if __name__ == '__main__':
    classifier = build_classifier(n_classes=1000, input_len=22050*3, stem='conv2d')
