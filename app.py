# import tensorflow
# from flask import Flask, request, jsonify, url_for
# import csv
# import math
# import os
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from werkzeug.utils import secure_filename
# import PIL
# import google.generativeai as genai
#
#
# # tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
# app = Flask(__name__, template_folder=tmpl_dir)
#
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
#
# os.environ['GOOGLE_API_KEY'] = "AIzaSyDrzwEvzqTTx33g3ikKoWyRU_mtxCuDa7s"
# genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
#
# # # define label meaning
# label = ['apple pie',
#          'baby back ribs',
#          'baklava',
#          'beef carpaccio',
#          'beef tartare',
#          'beet salad',
#          'beignets',
#          'bibimbap',
#          'bread pudding',
#          'breakfast burrito',
#          'bruschetta',
#          'caesar salad',
#          'cannoli',
#          'caprese salad',
#          'carrot cake',
#          'ceviche',
#          'cheese plate',
#          'cheesecake',
#          'chicken curry',
#          'chicken quesadilla',
#          'chicken wings',
#          'chocolate cake',
#          'chocolate mousse',
#          'churros',
#          'clam chowder',
#          'club sandwich',
#          'crab cakes',
#          'creme brulee',
#          'croque madame',
#          'cup cakes',
#          'deviled eggs',
#          'donuts',
#          'dumplings',
#          'edamame',
#          'eggs benedict',
#          'escargots',
#          'falafel',
#          'filet mignon',
#          'fish and_chips',
#          'foie gras',
#          'french fries',
#          'french onion soup',
#          'french toast',
#          'fried calamari',
#          'fried rice',
#          'frozen yogurt',
#          'garlic bread',
#          'gnocchi',
#          'greek salad',
#          'grilled cheese sandwich',
#          'grilled salmon',
#          'guacamole',
#          'gyoza',
#          'hamburger',
#          'hot and sour soup',
#          'hot dog',
#          'huevos rancheros',
#          'hummus',
#          'ice cream',
#          'lasagna',
#          'lobster bisque',
#          'lobster roll sandwich',
#          'macaroni and cheese',
#          'macarons',
#          'miso soup',
#          'mussels',
#          'nachos',
#          'omelette',
#          'onion rings',
#          'oysters',
#          'pad thai',
#          'paella',
#          'pancakes',
#          'panna cotta',
#          'peking duck',
#          'pho',
#          'pizza',
#          'pork chop',
#          'poutine',
#          'prime rib',
#          'pulled pork sandwich',
#          'ramen',
#          'ravioli',
#          'red velvet cake',
#          'risotto',
#          'samosa',
#          'sashimi',
#          'scallops',
#          'seaweed salad',
#          'shrimp and grits',
#          'spaghetti bolognese',
#          'spaghetti carbonara',
#          'spring rolls',
#          'steak',
#          'strawberry shortcake',
#          'sushi',
#          'tacos',
#          'octopus balls',
#          'tiramisu',
#          'tuna tartare',
#          'waffles']
#
# nu_link = 'https://www.nutritionix.com/food/'
#
# # Loading the best saved model to make predictions.
# tensorflow.keras.backend.clear_session()
# model_best = load_model('/home/div/Downloads/FoodCalorieEstimation-main/model_trained_101class.keras', compile=False)
# print('model successfully loaded!')
#
# start = [0]
# passed = [0]
# pack = [[]]
# num = [0]
#
# nutrients = [
#     {'name': 'protein', 'value': 0.0},
#     {'name': 'calcium', 'value': 0.0},
#     {'name': 'fat', 'value': 0.0},
#     {'name': 'carbohydrates', 'value': 0.0},
#     {'name': 'vitamins', 'value': 0.0}
# ]
#
# with open('nutrition101.csv', 'r') as file:
#     reader = csv.reader(file)
#     nutrition_table = dict()
#     for i, row in enumerate(reader):
#         if i == 0:
#             name = ''
#             continue
#         else:
#             name = row[1].strip()
#         nutrition_table[name] = [
#             {'name': 'protein', 'value': float(row[2])},
#             {'name': 'calcium', 'value': float(row[3])},
#             {'name': 'fat', 'value': float(row[4])},
#             {'name': 'carbohydrates', 'value': float(row[5])},
#             {'name': 'vitamins', 'value': float(row[6])}
#         ]
#
#
# # @app.route('/')
# # def index():
# #     img = 'static/profile.jpg'
# #     return render_template('index.html', img=img)
# #
# #
# # @app.route('/recognize')
# # def magic():
# #     return render_template('recognize.html', img=file)
#
#
# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     file = request.files.getlist("img")
# #     for f in file:
# #         filename = secure_filename(str(num[0] + 500) + '.jpg')
# #         num[0] += 1
# #         name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         print('save name', name)
# #         f.save(name)
# #
# #     pack[0] = []
# #     return render_template('recognize.html', img=file)
#
#
# @app.route('/predict')
# def predict():
#     result = []
#     # pack = []
#     print('total image', num[0])
#     for i in range(start[0], num[0]):
#         pa = dict()
#
#         filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
#         print('image filepath', filename)
#         pred_img = filename
#         pred_img = tensorflow.keras.preprocessing.image.load_img(pred_img, target_size=(224, 224))
#         pred_img = tensorflow.keras.preprocessing.image.img_to_array(pred_img)
#         pred_img = np.expand_dims(pred_img, axis=0)  # Add a batch dimension for prediction
#         pred_img = pred_img / 255.0  # Normalize pixel values (assuming they're between 0 and 255)
#
# # Make prediction
#         pred = model_best.predict(pred_img)
#         print("Pred")
#         print(pred)
#
#         if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
#                 math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
#             pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])
#
#         top = pred.argsort()[0][-3:]
#         label.sort()
#         _true = label[top[2]]
#         pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
#         x = dict()
#         x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
#         x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
#         x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))
#         pa['result'] = x
#         pa['nutrition'] = nutrition_table[_true]
#         pa['food'] = f'{nu_link}{_true}'
#         pa['idx'] = i - start[0]
#         pa['quantity'] = 100
#         img = PIL.Image.open(filename)
#         vision_model = genai.GenerativeModel('gemini-1.5-flash')
#         pa['ai'] = vision_model.generate_content(["Give me response in this form  'The image displays /items/ with an /estimated calories for all items and give accurate calories /'  ", img])
#
#         pack[0].append(pa)
#         passed[0] += 1
#
#     start[0] = passed[0]
#     print('successfully packed')
#
#     # compute the average source of calories
#     for p in pack[0]:
#         nutrients[0]['value'] = (nutrients[0]['value'] + p['nutrition'][0]['value'])
#         nutrients[1]['value'] = (nutrients[1]['value'] + p['nutrition'][1]['value'])
#         nutrients[2]['value'] = (nutrients[2]['value'] + p['nutrition'][2]['value'])
#         nutrients[3]['value'] = (nutrients[3]['value'] + p['nutrition'][3]['value'])
#         nutrients[4]['value'] = (nutrients[4]['value'] + p['nutrition'][4]['value'])
#
#     nutrients[0]['value'] = nutrients[0]['value'] / num[0]
#     nutrients[1]['value'] = nutrients[1]['value'] / num[0]
#     nutrients[2]['value'] = nutrients[2]['value'] / num[0]
#     nutrients[3]['value'] = nutrients[3]['value'] / num[0]
#     nutrients[4]['value'] = nutrients[4]['value'] / num[0]
#
#     response = {
#         "image_url": url_for('static', filename='uploads/' + filename, _external=True),
#         "predictions": predictions,
#         "gen_ai_text": gen_ai_text,
#         "pie_chart": pie_chart_base64,
#         "nutritionix_link": "https://www.nutritionix.com/"
#     }
#
#     # return render_template('results.html', pack=pack[0], whole_nutrition=nutrients)
#
# if __name__ == "__main__":
#     import click
#
#     @click.command()
#     @click.option('--debug', is_flag=True)
#     @click.option('--threaded', is_flag=True)
#     @click.argument('HOST', default='127.0.0.1')
#     @click.argument('PORT', default=5000, type=int)
#     def run(debug, threaded, host, port):
#         """
#         This function handles command line parameters.
#         Run the server using
#             python server.py
#         Show the help text using
#             python server.py --help
#         """
#         HOST, PORT = host, port
#         app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
#     run()


import tensorflow
from flask import Flask, request, jsonify
import csv
import math
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import PIL
import google.generativeai as genai

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.environ['GOOGLE_API_KEY'] = "AIzaSyDrzwEvzqTTx33g3ikKoWyRU_mtxCuDa7s"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Define food labels (food101 classes)
label = ['apple pie',
         'baby back ribs',
         'baklava',
         'beef carpaccio',
         'beef tartare',
         'beet salad',
         'beignets',
         'bibimbap',
         'bread pudding',
         'breakfast burrito',
         'bruschetta',
         'caesar salad',
         'cannoli',
         'caprese salad',
         'carrot cake',
         'ceviche',
         'cheese plate',
         'cheesecake',
         'chicken curry',
         'chicken quesadilla',
         'chicken wings',
         'chocolate cake',
         'chocolate mousse',
         'churros',
         'clam chowder',
         'club sandwich',
         'crab cakes',
         'creme brulee',
         'croque madame',
         'cup cakes',
         'deviled eggs',
         'donuts',
         'dumplings',
         'edamame',
         'eggs benedict',
         'escargots',
         'falafel',
         'filet mignon',
         'fish and_chips',
         'foie gras',
         'french fries',
         'french onion soup',
         'french toast',
         'fried calamari',
         'fried rice',
         'frozen yogurt',
         'garlic bread',
         'gnocchi',
         'greek salad',
         'grilled cheese sandwich',
         'grilled salmon',
         'guacamole',
         'gyoza',
         'hamburger',
         'hot and sour soup',
         'hot dog',
         'huevos rancheros',
         'hummus',
         'ice cream',
         'lasagna',
         'lobster bisque',
         'lobster roll sandwich',
         'macaroni and cheese',
         'macarons',
         'miso soup',
         'mussels',
         'nachos',
         'omelette',
         'onion rings',
         'oysters',
         'pad thai',
         'paella',
         'pancakes',
         'panna cotta',
         'peking duck',
         'pho',
         'pizza',
         'pork chop',
         'poutine',
         'prime rib',
         'pulled pork sandwich',
         'ramen',
         'ravioli',
         'red velvet cake',
         'risotto',
         'samosa',
         'sashimi',
         'scallops',
         'seaweed salad',
         'shrimp and grits',
         'spaghetti bolognese',
         'spaghetti carbonara',
         'spring rolls',
         'steak',
         'strawberry shortcake',
         'sushi',
         'tacos',
         'octopus balls',
         'tiramisu',
         'tuna tartare',
         'waffles']

# Nutrition link base URL
nu_link = 'https://www.nutritionix.com/food/'

# Load the pre-trained model
tensorflow.keras.backend.clear_session()

import os
import requests
from keras.models import load_model

# GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1cJt6ifr4aTdPj_R7sFJ86Kfz0b43HWFV"
# MODEL_PATH = "model_trained_101class.keras"
#
# def download_model():
#     """Download the model from Google Drive if not already present."""
#     if not os.path.exists(MODEL_PATH):
#         print("Downloading model...")
#         response = requests.get(GDRIVE_URL, stream=True)
#         with open(MODEL_PATH, "wb") as file:
#             for chunk in response.iter_content(chunk_size=1024):
#                 if chunk:
#                     file.write(chunk)
#         print("Download complete.")

# GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1cJt6ifr4aTdPj_R7sFJ86Kfz0b43HWFV"
# MODEL_PATH = "/tmp/model_trained_101class.keras"  # Store in the temporary directory
#
# def download_model():
#     """Download the model from Google Drive if not already present."""
#     if not os.path.exists(MODEL_PATH):
#         print("Downloading model...")
#         response = requests.get(GDRIVE_URL, stream=True)
#         with open(MODEL_PATH, "wb") as file:
#             for chunk in response.iter_content(chunk_size=1024):
#                 if chunk:
#                     file.write(chunk)
#         print("Download complete.")
#
# # Ensure model is available before loading
# download_model()
#
# # Load the model
# model_best = load_model(MODEL_PATH, compile=False)
# print("Model Loaded Successfully!")


# Define the Google Drive direct download link and model path
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1cJt6ifr4aTdPj_R7sFJ86Kfz0b43HWFV"
MODEL_PATH = "./tmp/model_trained_101class.keras"  # Store in the temporary directory


def download_model():
    """Download the model from Google Drive if not already present."""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model will be saved at: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")

        # Start a session to handle cookies
        session = requests.Session()
        response = session.get(GDRIVE_URL, stream=True)

        # Check for a Google Drive confirmation token (for large files)
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                GDRIVE_URL_with_token = GDRIVE_URL + "&confirm=" + value
                response = session.get(GDRIVE_URL_with_token, stream=True)

        # Download the file in chunks
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        print("Download complete.")
        if os.path.exists(MODEL_PATH):
            print(f"Current working directory: {os.getcwd()}")
            # print("Downloading model...")
    else:
        print("Model already exists.")


# Ensure model is available before loading
download_model()

# Load the model
model_best = load_model(MODEL_PATH, compile=False)
print("Model Loaded Successfully!")

# Load nutrition data from CSV file
nutrition_table = dict()
with open('nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i == 0:
            continue  # Skip header row
        name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]


@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Retrieve the single uploaded image
    file = request.files.get("img")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    # Save the uploaded image
    filename = secure_filename("uploaded.jpg")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image for prediction
    pred_img = tensorflow.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    pred_img = tensorflow.keras.preprocessing.image.img_to_array(pred_img)
    pred_img = np.expand_dims(pred_img, axis=0)  # Add batch dimension
    pred_img = pred_img / 255.0  # Normalize pixel values

    # Perform prediction with the pre-trained model
    pred = model_best.predict(pred_img)
    # Fallback for NaN predictions (if applicable)
    if (math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and
            math.isnan(pred[0][2]) and math.isnan(pred[0][3])):
        pred = np.array([[0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0]])

    # Get indices of top three predictions
    top = pred.argsort()[0][-3:]
    label.sort()  # Ensure the labels are sorted
    top_food = label[top[2]]  # Highest predicted label

    # Create a result dictionary for the predictions with probability percentages
    result = {
        top_food: float("{:.2f}".format(pred[0][top[2]] * 100)),
        label[top[1]]: float("{:.2f}".format(pred[0][top[1]] * 100)),
        label[top[0]]: float("{:.2f}".format(pred[0][top[0]] * 100))
    }

    # Retrieve nutrition data for the predicted food
    nutrition_data = nutrition_table.get(top_food, [])
    # Construct nutrition URL
    food_link = f'{nu_link}{top_food}'

    # Use generative AI to get a descriptive response (assuming synchronous response)
    img = PIL.Image.open(filepath)
    vision_model = genai.GenerativeModel('gemini-1.5-flash')
    ai_response = vision_model.generate_content([
        "Give me response in this form 'The image displays /items/ with an /estimated calories for all items and give accurate calories /' ",
        img
    ])

    # Prepare the API response data as a JSON object
    response_data = {
        "image": filepath,
        "result": result,
        "nutrition": nutrition_data,
        "food": food_link,
        "ai": ai_response.text  # Assumes the generative AI response has a 'text' attribute
    }

    return jsonify(response_data)


# @app.route('/')
# def home():
#     return "Flask App is Running on Render!"
#
# if __name__ == "__main__":
#     import click
#
#
#     @click.command()
#     @click.option('--debug', is_flag=True)
#     @click.option('--threaded', is_flag=True)
#     @click.argument('HOST', default='0.0.0.0')
#     @click.argument('PORT', default=5000, type=int)
#     def run(debug, threaded, host, port):
#         """
#         Run the server using:
#             python server.py
#         Use:
#             python server.py --help
#         to see help text.
#         """
#         HOST, PORT = host, port
#         app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
#
#
#     run()
@app.route("/")
def home():
    return "Flask App is Running on Render!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned PORT
    app.run(host="0.0.0.0", port=port, debug=True)
