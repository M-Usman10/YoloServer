import logging
from tools import *
from flask import send_from_directory, request,Flask
from werkzeug import secure_filename
from yolo import YOLO,process_video
from flask_cors import CORS


model = YOLO()

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def make_flask_app(receivingFolder):
    app = Flask(__name__)
    file_handler = logging.FileHandler('server.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    UPLOAD_FOLDER = receivingFolder
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    return app

app = make_flask_app("FlaskReceive")
CORS(app)

def detect_on_video(videoPath,returnPath):
    print(model)
    D = Data({'source': videoPath, 'step': 5, 'size': (416, 416, 3), 'video': 1})
    res_video = process_video(model, D.data, './Txt_results', D.names)
    boxes, names = read_boxes('Txt_results/boxes.txt')
    features, _ = read_boxes('Txt_results/features.txt', n=3)
    video = detectVideoCrowd(D.data, boxes, features, names)
    save_video(video,  returnPath)

@app.route('/DetectCrowd', methods = ['POST'])
def DetectCrowd():
    print ("request received")
    app.logger.info(app.config['UPLOAD_FOLDER'])
    video = request.files['FileName']
    video_name = secure_filename(video.filename)
    print ("Video/image to process is {}".format(video_name))
    create_new_folder(app.config['UPLOAD_FOLDER'])
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    app.logger.info("saving {}".format(saved_path))
    video.save(saved_path)
    detect_on_video(saved_path,saved_path)
    return send_from_directory(os.path.dirname(saved_path), os.path.basename(saved_path),as_attachment=True)

@app.route("/",methods=['POST','GET'])
def index_fn():
    return "Enter Valid Path"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9090, debug=False)