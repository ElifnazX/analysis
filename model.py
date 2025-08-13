import os #Dosya ve klasör işlemleri için kullanılır.
import shutil #Dosya kopyalama işlemleri için
from sklearn.model_selection import train_test_split #Veri setini eğitim ve doğrulama olarak ikiye ayırmak için 

# Orijinal veri seti klasör dizini
original_dataset_dir = 'C:/Users/Elifnaz Altınkaya/Desktop/analysis/dataset/train'
categories = ['covid', 'normal', 'viral_pneumonia']

# Yeni eğitim ve doğrulama dizinleri
base_dir = 'C:/Users/Elifnaz Altınkaya/Desktop/analysis/dataset'
train_dir = os.path.join(base_dir, 'train_split')
val_dir = os.path.join(base_dir, 'val_split')

#Verinin %80’i eğitim, %20’si doğrulama için ayrılacak
split_ratio = 0.8

for category in categories:
    # Klasör adını normalize et
    category_lower = category.lower().replace(' ', '_')

    # Yeni klasörleri oluştur
    os.makedirs(os.path.join(train_dir, category_lower), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category_lower), exist_ok=True)

    # images klasörüne git
    
    image_folder = os.path.join(original_dataset_dir, category, 'images')
    all_images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Veriyi ayır
    train_images, val_images = train_test_split(all_images, train_size=split_ratio, random_state=42)

    # Eğitim verilerini kopyala
    for img in train_images:
        src = os.path.join(image_folder, img)
        dst = os.path.join(train_dir, category_lower, img)
        shutil.copyfile(src, dst)

    # Doğrulama verilerini kopyala
    for img in val_images:
        src = os.path.join(image_folder, img)
        dst = os.path.join(val_dir, category_lower, img)
        shutil.copyfile(src, dst)

print("Veri bölme işlemi tamamlandı.")