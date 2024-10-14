import tensorflow as tf
import tensorflow.keras.saving
#import tensorflow_model_optimization as tfmot

model = tf.keras.saving.load_model("trained_model1.keras")
"""prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                             final_sparsity=0.90,
                                                             begin_step=0,
                                                             end_step=end_step)
}

model = prune_low_magnitude(model, **pruning_params)
model.compile(...)
model.fit(...)

model = tfmot.sparsity.keras.strip_pruning(model) """

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)