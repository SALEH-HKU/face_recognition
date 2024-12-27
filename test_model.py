import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, filedialog, Label, Button, Canvas, PhotoImage
from PIL import Image, ImageTk

# تحميل النموذج المدرب
model = load_model("student_recognition_model.h5")

# خريطة التصنيفات إلى الأسماء
label_to_name = {
    0: "Saleh Albarho",
    1: "Ahmad Alaatar",
    2: "Ghassan Churbaji",
    3: "Fares Sharabatli",
    4: "Abed Albaik",
    5: "Abdulkhader Albaik",
    6: "Giath Churbaji"
}

# إنشاء الواجهة
root = Tk()
root.title("Student Recognition")
root.geometry("600x400")

# منطقة لعرض الصورة
canvas = Canvas(root, width=224, height=224, bg="gray")
canvas.pack(pady=20)

# تسمية لعرض اسم الطالب
result_label = Label(root, text="Please upload an image", font=("Arial", 16))
result_label.pack(pady=10)

# دالة لتحميل الصورة والتنبؤ
def upload_and_predict():
    # فتح نافذة اختيار الصورة
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        try:
            # تحميل الصورة وعرضها على الواجهة
            img = Image.open(image_path)
            img_resized = img.resize((224, 224))
            img_tk = ImageTk.PhotoImage(img_resized)
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            canvas.image = img_tk  # الحفاظ على المرجع للصورة

            # معالجة الصورة للنموذج
            image_array = img_to_array(img_resized) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # التنبؤ بالصورة
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # عرض اسم الطالب
            student_name = label_to_name[predicted_class]
            result_label.config(text=f"Student: {student_name}")
        except Exception as e:
            result_label.config(text=f"Error: {e}")
    else:
        result_label.config(text="No image selected!")

# زر لرفع الصورة
upload_button = Button(root, text="Upload Image", command=upload_and_predict, font=("Arial", 14))
upload_button.pack(pady=10)

# تشغيل التطبيق
root.mainloop()
