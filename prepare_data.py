import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# إعداد مولد البيانات
data_gen = ImageDataGenerator(rescale=1./255)  # أزل خيار validation_split

# تحميل بيانات التدريب
train_data = data_gen.flow_from_directory(
    'dataset',  # مسار المجلد الرئيسي
    target_size=(224, 224),  # حجم الصور
    batch_size=16,          # حجم الدفعة (Batch Size)
    class_mode='categorical'
)

# عرض خريطة التصنيفات
print("Map of Labels to Names:", train_data.class_indices)
