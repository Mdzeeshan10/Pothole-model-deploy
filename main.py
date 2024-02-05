#flask utils
from flask import Flask, render_template, url_for, request, jsonify
from werkzeug.utils import secure_filename

upload_folder = '/uploads'
allowed_extentions = set(['png','jpg','jpeg'])
from keras.models import load_model
import cv2
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76

# azure computer image analyze import------------

# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from msrest.authentication import CognitiveServicesCredentials
# from python_code import vision

cog_key = 'f17b51a61a774e58b3445baa39549570'
cog_endpoint = 'https://ai-zee-computer.cognitiveservices.azure.com/'

#--------------------------------------------------------------------------------------



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extentions

app = Flask(__name__)
app.config['upload_folder'] =upload_folder

model = load_model('model/model_VGG16.h5')
def model_pred(input_img):
    img = cv2.imread(input_img)

    test_img = cv2.resize(img, (256,256))
    img_input = test_img.reshape((1,256,256,3))


    pred = model.predict(img_input)

    if pred[0] == 0:
        result = "Normal Road"

    else:
        result = "Pothol is Comming"

    return result

# color using in image parcentage show Model.

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors, autopct='%1.2f%%')

    
    return plt.show() 

#Azure image function define----------------------------------------------------------------------------------------------------
# def image_in_azure(image_path):
#     # # Get the path to an image file
#     # image_path = os.path.join('vission_data', im)

#     # Get a client for the computer vision service
#     computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

#     # Specify the features we want to analyze
#     features = ['Description', 'Tags', 'Adult', 'Objects', 'Faces']

#     # Get an analysis from the computer vision service
#     image_stream = open(image_path, "rb")
#     analysis = computervision_client.analyze_image_in_stream(image_stream, visual_features=features)

#     # Show the results of analysis (code in helper_scripts/vision.py)
#     result = vision.show_image_analysis(image_path, analysis)

#     return result



#Video capture code and its functions----------------------------------------------------

# def model_pred(input_img):
#     #img = cv2.imread(input_img)

#     test_img = cv2.resize(input_img, (256,256))
#     img_input = test_img.reshape((1,256,256,3))


#     pred = model.predict(img_input)

#     return pred


# def draw_label(img, text, pos, bg_color):
    
#     text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    
#     end_x = pos[0] + text_size[0][0] + 2
#     end_y = pos[0] + text_size[0][1] - 2
    
#     #cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
#     cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)


# cap = cv2.VideoCapture(0) 

# #----------------------------------------------------------------------------------------



@app.route("/", methods=['GET'])
def finalpage():
    return render_template('finalpage.html')

# calling Home Page


#calling Deplymodels Page

@app.route("/Deploymodels", methods=['GET'])
def Deploymodels():
    return render_template('Deploymodels.html')

@app.route('/upload', methods=['GET', 'POST'])

def upload_media():
    if 'file' not in request.files:
        return render_template('Deploymodels.html', result=(jsonify({'error': 'media not provided'}), 400))
    
    file = request.files['file']
    # filename =secure_filename(file.filename)
    # path =os.path.join('uploads', filename)
    # file.save(path)
    # re = model_pred(path)
    # return re
    

    if file.filename == '':
        return render_template('Deploymodels.html', result='no file selected')
    if file and allowed_file(file.filename):
        filename =secure_filename(file.filename)
        
        path =os.path.join('uploads', filename)
        file.save(path)
        re = model_pred(path)
        return render_template('Deploymodels.html', result=re)
    return render_template('Deploymodels.html', result='correct predict')

#Video start and and live detect pothole
@app.route('/video', methods=['GET', 'POST'])

def video_detect():
    #Video capture code and its functions----------------------------------------------------

    def model_pred(input_img):
        #img = cv2.imread(input_img)

        test_img = cv2.resize(input_img, (256,256))
        img_input = test_img.reshape((1,256,256,3))


        pred = model.predict(img_input)

        return pred


    def draw_label(img, text, pos, bg_color):
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
        
        end_x = pos[0] + text_size[0][0] + 2
        end_y = pos[0] + text_size[0][1] - 2
        
        #cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)


    cap = cv2.VideoCapture(0) 

    #----------------------------------------------------------------------------------------

    while cap.isOpened():
        rat, frame = cap.read()
        
        pred = model_pred(frame)
        
        if pred[0] == 0:
            draw_label(frame, "Normal Road", (30,30), (225,225,0))
            

        else:
            draw_label(frame, "Pothol is Comming", (30,30), (225,225,0))
        
            
        
        cv2.imshow("window",frame)
        
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break
    cap.release()
    cv2.destroyAllWindows()



    # print(model_pred(path))

    # if request.method == 'POST':
    #     #Get the file from post request
    #     f = request.files['file']
    #     #save the file to ./uploads
    #     # basepath = os.path.dirname(__file__)
    #     # file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    #     # f.save(file_path)
    #     model_pred(f)


# color using in image parcentage show.
@app.route("/colorimageshow", methods=['GET'])
def colorimageshow():
    return render_template('colordetectiondeploy.html')

@app.route('/show', methods=['GET', 'POST'])

def show_color():
    if 'file' not in request.files:
        return render_template('colordetectiondeploy.html', result=(jsonify({'error': 'media not provided'}), 400))
    
    file = request.files['file']
    # filename =secure_filename(file.filename)
    # path =os.path.join('uploads', filename)
    # file.save(path)
    # re = model_pred(path)
    # return re
    

    if file.filename == '':
        return render_template('colordetectiondeploy.html', result='no file selected')
    if file and allowed_file(file.filename):
        filename =secure_filename(file.filename)
        nocolor = int(request.form.get('nocolor'))
        
        path =os.path.join('uploads', filename)
        file.save(path)
        re = get_colors(get_image(path), nocolor, True)
        return render_template('colordetectiondeploy.html', result=re)
    return render_template('colordetectiondeploy.htmll', result='correct predict')

# Azure Computer Vission To apply code ---------------------------------------------------------------------------------------

# color using in image parcentage show.
@app.route("/analyzecomputervission", methods=['GET'])
def analyzecomputervission():
    return render_template('deplycomputervission.html')

# @app.route('/imageanalyze', methods=['GET', 'POST'])

# def azure_image_analyze():
#     if 'file' not in request.files:
#         return render_template('deplycomputervission.html', result=(jsonify({'error': 'media not provided'}), 400))
    
#     file = request.files['file']
#     # filename =secure_filename(file.filename)
#     # path =os.path.join('uploads', filename)
#     # file.save(path)
#     # re = model_pred(path)
#     # return re
    

#     if file.filename == '':
#         return render_template('deplycomputervission.html', result='no file selected')
#     if file and allowed_file(file.filename):
#         filename =secure_filename(file.filename)
        
#         path =os.path.join('uploads', filename)
#         file.save(path)
#         re = image_in_azure(path)
#         return render_template('deplycomputervission.html', result=re)
#     return render_template('deplycomputervission.htmll', result='correct predict')






if __name__=='__main__':
    app.run(debug=False)