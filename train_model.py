import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# إعداد مولد البيانات
data_gen = ImageDataGenerator(rescale=1./255)

# تحميل بيانات التدريب
train_data = data_gen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# تحميل MobileNetV2 كنموذج أساسي
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # تثبيت طبقات MobileNetV2

# إنشاء النموذج النهائي
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')  # عدد الفئات يساوي عدد الطلاب
])

# تجميع النموذج
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(train_data, epochs=10)

# حفظ النموذج
model.save("student_recognition_model.h5")
print("Model trained and saved successfully!")
