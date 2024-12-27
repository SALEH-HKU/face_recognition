import tensorflow as tf

# تحميل النموذج المدرب
model = tf.keras.models.load_model("student_recognition_model.h5")

# تحويل النموذج إلى TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# تحسين النموذج (اختياري: تحسين الحجم والأداء)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# حفظ النموذج المحول
tflite_model = converter.convert()
with open("student_recognition_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite!")
