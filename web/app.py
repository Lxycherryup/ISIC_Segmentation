import os
import cv2
import numpy as np

from flask import Flask,render_template, url_for,request,redirect,jsonify,session
from werkzeug.utils import secure_filename

from scipy import ndimage as ndi
from PIL import  Image
from threading import Thread
from flask_cors import CORS
from test_one_image import predict_single_image

"""
Flask是一个使用 Python 编写的轻量级 Web 应用框架，也被称为 “微服务框架” ，因为使用简单的核心（ Werkzeug WSGI工具包2、 Jinja2 模板引擎3）且支持插件扩展。
"""

# Flask类是Flask框架核心，实现了WSGI规范

#创建Flask实例对象
"""
创建了一个名为app的Flask应用程序实例，以当前模板的名称作为应用程序的根目录。
后续可以使用这个实例来配置路由、注册插件、定义视图函数等，从而构建一个完整的Web应用程序
"""
app = Flask(__name__)
CORS(app)  #允许跨域请求
app.secret_key = '123456'  # 设置一个安全密钥


#--------------------------------------搭建路由------------------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template('index.html')  #render_template('index.html',参数1,参数2)传的参数  可以在{{参数}}的形式 在前端页面中渲染出来

@app.route('/return_home_page',methods = ['GET', 'POST'])
def return_home_page():
    return render_template('index.html')

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # 检查是否有文件在请求中
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        # 如果用户没有选择文件，浏览器也可能提交一个空的文件无文件名。
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            # 保存文件到static目录
            save_path =os.path.join('static/image',filename)
            file.save(save_path)

            # 分割结果保存地址
            predicet_save_path = save_path.replace('image','pred')
            mask_path = '../Data/ISIC2018/TestDataset/test/masks/' + filename

            # 进行分割
            metrics = predict_single_image(image_path=save_path,mask_path = mask_path)
            jaccard = metrics[0]
            f1 = metrics[1]
            recall = metrics[2]
            precision = metrics[3]
            acc = metrics[4]
            f2 = metrics[5]
            res = f'Jaccard:{jaccard:.4f},Dice:{f1:.4f},Recall:{recall:.4f},Precision:{precision:.4f},Accuracy:{acc:.4f},F2:{f2:.4f}'
            return render_template('upload.html',save_path=save_path,predicet_save_path=predicet_save_path,res = res)


if __name__ =="__main__":
    app.run(debug=True)

