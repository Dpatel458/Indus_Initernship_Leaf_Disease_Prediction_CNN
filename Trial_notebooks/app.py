from flask import Flask, render_template , request
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import os, random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


data_dir = r"C:\Users\Ishan\Internship project\dataset\plantvillage dataset\color"
classes = os.listdir(data_dir)
class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model("updated_resnet100.h5")

model2 = load_model("fine_tune_100_percent_resnet20.h5")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_with_loaded_model(image_path, model, class_names):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', error="No image selected")

        image = request.files['image']
        if image.filename == '':
            return render_template('predict.html', error="No image selected")

        filename = "uploaded_image.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Predict
        predicted_class, confidence = predict_with_loaded_model(image_path, model, class_names)
        predicted_class2, confidence2 = predict_with_loaded_model(image_path, model2, class_names)
        return render_template('predict.html',
                               image_url=image_path,
                               prediction=predicted_class,
                               confidence=f"{confidence*100:.2f}%",
                               prediction2=predicted_class2,
                               confidence2=f"{confidence2*100:.2f}%")

        

    return render_template("predict.html")


@app.route('/visuals')
def visuals():
    return render_template("visuals.html")  # Create this file in /templates

@app.route('/analysis')
def analysis():

#    os.makedirs("static/images/plots", exist_ok=True)

#     df = pd.DataFrame(list(class_counts.items()), columns=["class", "count"])
#     df[['crop', 'disease']] = df['class'].str.split('___', expand=True)
#     df['disease'] = df['disease'].fillna("Healthy")

#     # Plot 1: Images per class
#     plt.figure(figsize=(12, 8))
#     sns.barplot(data=df, x="class", y="count", palette="Blues_d")
#     plt.xticks(rotation=90)
#     plt.title("Number of Images per Class")
#     plt.tight_layout()
#     plt.savefig("static/images/plots/images_per_class.png")
#     plt.close()

#     # Plot 2: Images per crop
#     crop_counts = df.groupby('crop')['count'].sum().reset_index().sort_values(by='count', ascending=False)
#     plt.figure(figsize=(12, 8))
#     sns.barplot(data=crop_counts, x='crop', y='count', color='green')
#     plt.title("Total Images per Crop")
#     plt.xlabel("Crop Type")
#     plt.ylabel("Total Image Count")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("static/images/plots/images_per_crop.png")
#     plt.close()

#     # Plot 3: Images per disease
#     disease_counts = defaultdict(int)
#     for cls in df["class"]:
#         disease = cls.split("___")[1] if "___" in cls else "Healthy"
#         disease_counts[disease] += class_counts[cls]

#     df_diseases = pd.DataFrame(list(disease_counts.items()), columns=['Disease', 'Image Count'])
#     plt.figure(figsize=(12, 8))
#     sns.barplot(data=df_diseases, x='Disease', y='Image Count', color='orange')
#     plt.xticks(rotation=90)
#     plt.title("Total Images per Disease Type")
#     plt.tight_layout()
#     plt.savefig("static/images/plots/images_per_disease.png")
#     plt.close()

#     # Plot 4: Crop vs Disease heatmap
#     pivot_df = df.pivot_table(index='crop', columns='disease', values='count', fill_value=0)
#     plt.figure(figsize=(15, 8))
#     sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu")
#     plt.title("Crop vs Disease Image Count Heatmap")
#     plt.xlabel("Disease Type")
#     plt.ylabel("Crop")
#     plt.tight_layout()
#     plt.savefig("static/images/plots/crop_disease_heatmap.png")
#     plt.close()

#     # Plot 5: Healthy vs Diseased
#     healthy = sum(count for cls, count in class_counts.items() if 'healthy' in cls.lower())
#     diseased = sum(class_counts.values()) - healthy

#     data_health = pd.DataFrame({
#         'Condition': ['Healthy', 'Diseased'],
#         'Image Count': [healthy, diseased]
#     })

#     plt.figure(figsize=(12, 8))
#     sns.barplot(data=data_health, x='Condition', y='Image Count', hue='Condition',
#             palette={'Healthy': 'green', 'Diseased': 'red'}, legend=False)

#     plt.title("Healthy vs Diseased Leaf Images")
#     plt.ylabel("Image Count")
#     plt.tight_layout()
#     plt.savefig("static/images/plots/healthy_vs_diseased.png")
#     plt.close() 

    # plot_paths=[
    #     "static/images/plots/images_per_class.png",
    #     "static/images/plots/images_per_crop.png",
    #     "static/images/plots/images_per_disease.png",
    #     "static/images/plots/crop_disease_heatmap.png",
    #     "static/images/plots/healthy_vs_diseased.png",   
    #     "sstatic/images/plots/Image_Dimension_Distribution.png",
    #     "static/images/plots/Aspect_ratio_Distribution.png",
    #     "static/images/plots/Average_Brightness_Per_Class.png",
    #     "static/images/plots/RGB_Color_Channel_Histogram.png",
    #     "static/images/plots/RGB_Color_Histogram_For_a_Sample_Image.png"
    # ]
    

    return render_template("analysis.html")

@app.route('/metric')
def metric():
    return render_template("metric.html")


@app.route('/augmentation', methods=['GET', 'POST'])
def augmentation():
    original_path = "static/images/augmentation/original.png"
    augmented_path = "static/images/augmentation/augmented.png"
    class_name = None

    if request.method == 'POST':
        # Re-run augmentation on button click
        base_dir = "dataset/plantvillage dataset/color"
        class_names = [cls for cls in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, cls))]
        target_class = random.choice(class_names)
        target_dir = os.path.join(base_dir, target_class)
        random_image = random.choice(os.listdir(target_dir))
        random_image_path = os.path.join(target_dir, random_image)

        # Load image
        img = mpimg.imread(random_image_path)

        # Augmentation layer
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.4),
            tf.keras.layers.RandomZoom(0.4),
            tf.keras.layers.RandomTranslation(height_factor=0.3, width_factor=0.3),
           
        ])

        # Apply augmentation
        augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
        augmented_img = tf.squeeze(augmented_img).numpy()

        # Create output directory
        os.makedirs("static/images/augmentation", exist_ok=True)

        # Save images
        plt.imsave(original_path, img)
        plt.imsave(augmented_path, augmented_img / 255.0)
        class_name = target_class

    return render_template("augmentation.html",
                           original_img=original_path,
                           augmented_img=augmented_path,
                           class_name=class_name)

@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
