# -*- coding: UTF-8 -*-
import os
import weaviate
import sys
import base64
import shutil
import re
import numpy as np
import random
import io
import time
import cv2
import tkinter.messagebox as msgbox
from PyQt5.QtCore import *  # 此模块用于处理时间、文件和目录、各种数据类型、流、URL、MIME类型、线程或进程
from io import BytesIO
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageQt
from pathlib import Path
import matplotlib.pyplot as plt

from ui_ImageBrowserWidget1 import Ui_Form

"显示多张图片的缩略图 加滚动条"
FrameIdxRole = Qt.UserRole + 1

#Connect to Weaviate
print("Connect to Weaviate...")
client = weaviate.Client(
    url="http://localhost:8080",
)


#client = ""
def image_to_base64():
    """
    Convert images to base64
    """
    img_path = Path("./img/")
    b64_path = Path("b64_img")

    for file_path in img_path.glob("*"):
        if file_path.is_file():
            filename = file_path.name
            try:
                with open(file_path, "rb") as f:
                    image_data = f.read()
                b64_data = base64.b64encode(image_data).decode("utf-8")
                with open(b64_path / f"{filename}.b64", "w") as f:
                    f.write(b64_data)
            except Exception as e:
                print(f"Error converting {filename}: {e}")

def set_up_batch():
    """
    Set batch processing configuration parameters to accelerate the speed of importing and deleting data.
    """

    client.batch.configure(
        batch_size=100,
        dynamic=True,
        timeout_retries=3,
        callback=None,
    )

def clear_up_MyImages():
    """
    Remove all objects from the MyImages collection.
    This is useful if we want to rerun the import with different pictures.
    """
    try:
        with client.batch as batch:
            batch.delete_objects(
                class_name="MyImages",
                # same where operator as in the GraphQL API
                where={"operator": "NotEqual", "path": ["text"], "valueString": "x"},
                output="verbose",
            )
        print("All objects from ftd collection have been deleted.")
    except Exception as e:
        print(f"Error while clearing ftd collection: {e}")

def import_data():
    """
    Process all images in the [base64_images] folder and import them into the MyImages collection.
    """
    base64_images_folder = "./b64_img"
    search_string = 'negtive_'
    try:
        with client.batch as batch:
            num_files = len(os.listdir(base64_images_folder))
            for idx, encoded_file_path in enumerate(os.listdir(base64_images_folder), 1):
                with open(os.path.join(base64_images_folder, encoded_file_path)) as file:
                    file_lines = file.readlines()
                imageLabel = '0'  # 0代表正样本数据，1代表负样本数据
                base64_encoding = " ".join(file_lines).replace("\n", "").replace(" ", "")

                # remove .b64 to get the original file name
                image_file = encoded_file_path.replace(".b64", "")

                if search_string in image_file:
                    print(f'{image_file} 包含了字符串 {search_string}')
                    imageLabel = '1'

                # remove image file extension and swap - for " " to get the breed name
                breed = re.sub(r".(jpg|jpeg|png)", "", image_file).replace("-", " ")

                # The properties from our schema
                data_properties = {"image": base64_encoding, "text": imageLabel}
                print(image_file)
                print(imageLabel)

                batch.add_data_object(data_properties, "MyImages")

                # Progress tracking
                print(f"Processed file {idx}/{num_files} - {image_file}")

        print("All objects have been uploaded to Weaviate.")
    except Exception as e:
        print(f"Error while importing data: {e}")

# Function to search for similar images in Weaviate
def search_similar_images(img_str,max_distance):
    sourceImage = {"image": img_str}

    weaviate_results = client.query.get(
        "MyImages", ["text", "image"]
    ).with_near_image({"image": img_str,"distance": max_distance}, encode=False).with_additional(["distance"]).do()

    return weaviate_results.get("data", {}).get("Get", {}).get("MyImages", [])



class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)

        self.setupUi(self)

        self.pushButton.clicked.connect(self.display)


    def display(self):
        self.listWidgetImages.setViewMode(QListView.IconMode)
        self.listWidgetImages.setModelColumn(1)
        self.listWidgetImages.itemSelectionChanged.connect(self.onItemSelectionChanged)
        # slider
        i = 0
        self.sliderScale.valueChanged.connect(self.onSliderPosChanged)
        for filename in os.listdir(r"./photo"):
            i = i + 1
        # for i in range(100):
            image = cv2.imread("./photo" + "/" + filename)
            self.add_image_thumbnail(image,"result",str(i))


    def add_image_thumbnail(self, image, frameIdx, name):
        self.listWidgetImages.itemSelectionChanged.disconnect(self.onItemSelectionChanged)

        height, width, channels = image.shape
        print(image.shape)
        bytes_per_line = width * channels
        print(bytes_per_line)
        qImage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImage)

        item = QListWidgetItem(QIcon(pixmap), str(frameIdx) + ": " + name)
        item.setData(FrameIdxRole, frameIdx)

        self.listWidgetImages.addItem(item)

        # to bottom
        # self.listWidgetImages.scrollToBottom()
        self.listWidgetImages.setCurrentRow(self.listWidgetImages.count() - 1)

        print('\033[32;0m  --- add image thumbnail: {}, {} -------'.format(frameIdx, name))

        self.listWidgetImages.itemSelectionChanged.connect(self.onItemSelectionChanged)
        # self.listWidgetImages.it

    def resizeEvent(self, event):
        width = self.listWidgetImages.contentsRect().width()
        self.sliderScale.setMaximum(width)
        self.sliderScale.setValue(width - 40)

    def onItemSelectionChanged(self):
        pass

    def onSliderPosChanged(self, value):
        self.listWidgetImages.setIconSize(QSize(value, value))


class mainwindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()



    def init_ui(self):

        self.ui = uic.loadUi("./mainwindow.ui")  # 加载由Qt Designer设计的ui文件
        self.ui.setWindowTitle("Weaviate Database")
        self.ui.setStyleSheet("#MainWindow{background-color:#ffe5b4}")

        self.ui.label.setStyleSheet("border-width: 1px;border-style: solid;boder-color: rgb(0,0,0);")
        self.ui.label_4.setStyleSheet("border-width: 1px;border-style: solid;boder-color: rgb(0,0,0);")
        self.ui.label888.setStyleSheet("border-width: 1px;border-style: solid;boder-color: rgb(0,0,0);")
        self.ui.label_6.setStyleSheet("border-width: 1px;border-style: solid;boder-color: rgb(0,0,0);")

        self.ui.defineSchema.setStyleSheet("QPushButton{\n"
                                      "    background:#ffcc00 ;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px;font-size:24px;border-radius: 24px;font-family: Times New Roman;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    background:black;\n"
                                      "}")

        self.ui.convertBase64.setStyleSheet("QPushButton{\n"
                                      "    background:#ffcc00;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px;font-size:24px;border-radius: 24px;font-family: Times New Roman;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    background:black;\n"
                                      "}")

        self.ui.createDatabase.setStyleSheet("QPushButton{\n"
                                      "    background:#ffcc00;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px;font-size:24px;border-radius: 24px;font-family: Times New Roman;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    background:black;\n"
                                      "}")



        self.ui.similaritySearch.setStyleSheet("QPushButton{\n"
                                      "    background:#ffcc00;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px;font-size:28px;border-radius: 24px;font-family: Times New Roman;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    background:black;\n"
                                      "}")
        self.ui.PR_cal.setStyleSheet("QPushButton{\n"
                                      "    background:#ffcc00;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px;font-size:28px;border-radius: 24px;font-family: Times New Roman;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    background:black;\n"
                                      "}")


        # defineSchema 绑定槽函数 defineSchema()
        self.ui.defineSchema.clicked.connect(self.defineSchema)
        # convertBase64 绑定槽函数 convertBase64()
        self.ui.convertBase64.clicked.connect(self.convertBase64)
        # createDatabase 绑定槽函数 createDatabase()
        self.ui.createDatabase.clicked.connect(self.createDatabase)
        # similaritySearch 绑定槽函数 similaritySearch()
        self.ui.similaritySearch.clicked.connect(self.similaritySearch)
        # PR_cal 绑定槽函数 PR_cal()
        self.ui.PR_cal.clicked.connect(self.PR_cal)

    def printfGUI(self, mes):
        self.ui.textBrowser.append(mes)  # 在指定的区域显示提示信息
        self.cursot = self.ui.textBrowser.textCursor()
        self.ui.textBrowser.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents()



    def defineSchema(self):
        schemaConfig = {
            'class': 'MyImages',
            # class name for schema config in Weaviate (change it with a custom name for your images)
            'vectorizer': 'img2vec-neural',
            'vectorIndexType': 'hnsw',
            'moduleConfig': {
                'img2vec-neural': {
                    'imageFields': [
                        'image'
                    ]
                }
            },
            'properties': [
                {
                    'name': 'image',
                    'dataType': ['blob']
                },
                {
                    'name': 'text',
                    'dataType': ['string']
                }
            ]
        }

        try:
            client.schema.create_class(schemaConfig)
            print("Schema configured")
            self.printfGUI("Schema configured")
        except Exception:
            print("Schema already configured, skipping...")
            self.printfGUI("Schema already configured, skipping...")

    def convertBase64(self):
        start_time = time.time()
        print('ftd image convertBase64 running...')
        self.printfGUI("ftd image convertBase64 running...")
        base_folder = "b64_img"
        if os.path.exists(base_folder):
            shutil.rmtree(base_folder)
        os.mkdir(base_folder)
        image_to_base64()

        # 获取代码执行后时间戳
        end_time = time.time()
        # 计算执行时间
        run_time = end_time - start_time
        print("convertBase64 Done!")
        self.printfGUI("convertBase64 Done!")
        self.printfGUI("Image data conversion base64 time-consuming：{:.2f}  seconds".format(run_time))

    def createDatabase(self):

        start_time = time.time()
        self.printfGUI("Setting up batch processing...")
        # Weaviate Client Setup
        print("Setting up batch processing...")
        set_up_batch()
        # Clearing existing data in the collection
        print("Clearing existing data in ftd collection...")
        self.printfGUI("Clearing existing data in ftd collection...")
        clear_up_MyImages()
        # Import data into Weaviate
        print("Importing data into ftd collection...")
        self.printfGUI("Importing data into ftd collection...")
        start_time = time.time()
        import_data()
        # 获取代码执行后时间戳
        end_time = time.time()
        # 计算执行时间
        run_time = end_time - start_time
        self.printfGUI("Importing data into ftd collection Done!")
        self.printfGUI("Importing data into ftd collection time-consuming：{:.2f}  seconds".format(run_time))

    def similaritySearch(self):
        # 弹窗导入图片文件
        '''
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "inference/images", "*.jpg;;*.png;;All Files(*)")
        self.sourceName = img_name
        print('openFile start')
        image = Image.open(img_name)
        # Image转换成QImage
        qimage = ImageQt.toqimage(image)
        # 这里直接转成QPixmap，就可以直接使用了
        qpixmap = ImageQt.toqpixmap(image)
        self.ui.label.setPixmap(qpixmap)
        self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小
        '''

        uploaded_img, _ = QtWidgets.QFileDialog.getOpenFileName(self, "openFile", "inference/images", "*.png;;*.jpg;;All Files(*)")

        self.sourceName = uploaded_img
        print('openFile start')
        '''
        image = Image.open(uploaded_img)
        # Image转换成QImage
        qimage = ImageQt.toqimage(image)
        # 这里直接转成QPixmap，就可以直接使用了
        qpixmap = ImageQt.toqpixmap(image)
        self.ui.label.setPixmap(qpixmap)
        self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小
        '''

        pix = QtGui.QPixmap(uploaded_img)
        self.ui.label.setPixmap(pix)
        self.ui.label.setScaledContents(True)


        start_time = time.time()
        self.printfGUI("Start searching for similar images...")
        if uploaded_img is not None:
            # Display the uploaded image
            img_pil = Image.open(uploaded_img)

            # Convert the uploaded image to a base64 string
            with BytesIO() as output:
                img_pil.save(output, format="PNG")
                img_str = base64.b64encode(output.getvalue()).decode("utf-8")


            # similar_images = search_similar_images(img_str,2.1)
            # resultLabel_list = [result.get("text") for result in similar_images]
            # Search for similar images
            # max_distance = 0.0

            max_distance = 0.2
            similar_images = search_similar_images(img_str,max_distance)
            # 获取代码执行后时间戳
            end_time = time.time()
            # 计算执行时间
            run_time = end_time - start_time
            self.printfGUI("Search time-consuming：{:.2f}  seconds".format(run_time))

            if not similar_images:
                print("No similar images found in the database. Please upload something else.")
                self.printfGUI("No similar images found in the database. Please upload something else.")
            else:
                '''
                images = [base64.b64decode(result.get("image")) for result in similar_images]
                image0 = io.BytesIO(images[0])
                image1 = io.BytesIO(images[1])
                img0 = Image.open(image0)
                img1 = Image.open(image1)
                # Image转换成QImage
                qimage0 = ImageQt.toqimage(img0)
                qimage1 = ImageQt.toqimage(img1)
                # 这里直接转成QPixmap，就可以直接使用了
                qpixmap0 = ImageQt.toqpixmap(img0)
                qpixmap1 = ImageQt.toqpixmap(img1)
                self.ui.label_4.setPixmap(qpixmap0)
                self.ui.label888.setPixmap(qpixmap1)
                self.ui.label_4.setScaledContents(True)  # 设置图像自适应界面大小
                self.ui.label888.setScaledContents(True)  # 设置图像自适应界面大小
                '''
                images = [base64.b64decode(result.get("image")) for result in similar_images]
                #image0 = io.BytesIO(images[0])
                #image1 = io.BytesIO(images[1])
                save_img = "./photo/"
                shutil.rmtree(save_img)
                print('delte')
                os.mkdir(save_img)
                for ix in range(0,len(images)):
                    img_name = save_img + "result_" + str(ix) + ".png"
                    file = open(img_name, "wb")
                    file.write(images[ix])

                txt_lst = [similar_images[i]['text'] for i in range(2)]

                pix = QPixmap()
                pix.loadFromData(images[0])
                self.ui.label_4.setPixmap(pix)
                self.ui.label_4.setScaledContents(True)  # 设置图像自适应界面大小

                pix2 = QPixmap()
                pix2.loadFromData(images[1])
                self.ui.label888.setPixmap(pix2)
                self.ui.label888.setScaledContents(True)  # 设置图像自适应界面大小


               # self.ui.label888.setPixmap(QtGui.QPixmap.fromImage(QtImg2))
               # self.ui.label888.setScaledContents(True)  # 设置图像自适应界面大小


                '''
                img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # 转换灰度图
                print('opencv bgr: ', img_raw.shape)
                print('opencv gray: ', img_gray.shape)

                cv2.imshow("img bgr", img_raw)
                cv2.imshow("img gray", img_gray)
                cv2.waitKey(0)
                '''


    def PR_cal(self):
        uploaded_img, _ = QtWidgets.QFileDialog.getOpenFileName(self, "openFile", "inference/images", "*.png;;*.jpg;;All Files(*)")

        self.sourceName = uploaded_img
        print('openFile start')
        pix = QtGui.QPixmap(uploaded_img)
        self.ui.label.setPixmap(pix)
        self.ui.label.setScaledContents(True)
        #上传的正样本图片总数
        positiveSamples = 461
        negtiveSamples = 3188

        start_time = time.time()
        if uploaded_img is not None:
            # Display the uploaded image
            img_pil = Image.open(uploaded_img)

            # Convert the uploaded image to a base64 string
            with BytesIO() as output:
                img_pil.save(output, format="PNG")
                img_str = base64.b64encode(output.getvalue()).decode("utf-8")


            # similar_images = search_similar_images(img_str,2.1)
            # resultLabel_list = [result.get("text") for result in similar_images]
            # Search for similar images
            # max_distance = 0.0
            presionAll = []
            recallAll = []
            max_distanceAll = []
            data_list = []
            for i in range(0, 21):
                print(i)
                max_distance = float(i) / 20
                max_distanceAll.append(max_distance)
                similar_images = search_similar_images(img_str,max_distance)
                for result in  similar_images:
                    data_list.append({'image': result.get("image"), 'label': result.get("text")})
                resultLabel_list = [result.get("text") for result in similar_images]
                print(resultLabel_list)
                y_pred = list(map(int, resultLabel_list))
                print(y_pred)
                # 0代表正样本数据，1代表负样本数据
                # Precison=检索到的相似图片个数/检索到的全部图片总数=预测为正样本的图片个数/所有被预测为正样本的图片总数
                presion = np.sum(np.equal(y_pred, 0)) / len(y_pred)
                print(len(y_pred))
                print(presion)
                presionAll.append(presion)
                # Recall=检索到的相似图片个数/数据集里全部相似图片总数=预测为正样本的图片个数/所有正样本的图片总数
                recall = np.sum(np.equal(y_pred, 0)) / positiveSamples
                print(recall)
                recallAll.append(recall)
            self.save_img(data_list)
            # 获取代码执行后时间戳
            end_time = time.time()
            print(presionAll)
            print(recallAll)
            plt.plot(max_distanceAll, presionAll, label='Precision')
            plt.plot(max_distanceAll, recallAll, label='Recall')
            plt.xlabel('Distance threshold')
            # 添加图例
            plt.legend()
            plt.show()
            buffer = io.BytesIO()
            plt.savefig('temp.png')
            plt.close()
            self.ui.label_6.setPixmap(QtGui.QPixmap('temp.png'))

            # 计算执行时间
            run_time = end_time - start_time

    def save_img(self, data_list):
        import base64
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image
        from io import BytesIO

        # 创建一个新的 Excel 工作簿和工作表
        wb = Workbook()
        ws = wb.active

        # 遍历每个字典并将 base64 编码的图片和标签插入到 Excel 工作表中
        for index, data_dict in enumerate(data_list):
            try:
                # 获取 'imgs' 和 'label' 键对应的值
                image_base64 = data_dict.get('image', '')
                label = data_dict.get('label', '')

                # 解码 base64
                image_data = base64.b64decode(image_base64)

                # 创建 BytesIO 对象
                image_io = BytesIO(image_data)

                # 创建 Image 对象
                img = Image(image_io)

                # 计算图片在 Excel 中的大小和位置（可以根据需要调整）
                img.width = 100
                img.height = 100
                img.anchor = f'A{index + 2}'  # 假设图片插入到 A 列，并且从第二行开始

                # 将图片插入到 Excel 工作表中
                ws.add_image(img)

                # 将标签插入到 Excel 工作表中的 B 列（可以根据需要调整）
                ws.cell(row=index + 2, column=2, value=label)
                # 将表格的列宽和行高设置为图片的大小
                ws.column_dimensions[f'A'].width = img.width  # 调整比例以适应实际情况
                ws.row_dimensions[index + 2].height = img.height  # 调整比例以适应实际情况

                print(f"Image {index + 1} and label inserted successfully.")
            except Exception as e:
                print(f"Error inserting image {index + 1} and label: {e}")

        # 保存 Excel 文件（可以根据需要调整文件名和路径）
        wb.save('数据.xlsx')


class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Welcome to login')  # Set Title
        #self.resize(500, 500)  # 设置宽、高
        #self.set(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)  # Set the button to hide and close X

        # 设置背景为透明
        self.setStyleSheet('background-image:url("33.png")')
        '''
        Define interface control settings
        '''
        self.frame = QFrame(self)  # Initialize Frame Object
        self.frame.setGeometry(100, 100, 400, 300)



        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout

        self.login_id = QLineEdit()
        self.login_id.setPlaceholderText("Please enter your login account")
        self.login_id.setStyleSheet("background:#ffcc00 ;\n"   "    color:black;\n")
        self.verticalLayout.addWidget(self.login_id)  # Add the login account settings to the page control

        self.passwd = QLineEdit()
        self.passwd.setPlaceholderText("Please enter your login password")
        self.passwd.setStyleSheet("background:#ffcc00 ;\n"   "    color:black;\n")
        self.verticalLayout.addWidget(self.passwd)  # Add the login password setting to the page control

        self.button_enter = QPushButton()
        self.button_enter.setText("Login")
        self.button_enter.setStyleSheet("QPushButton{\n"
                                      "    background:#ffcc00;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px;font-size:24px;border-radius: 24px;font-family: Times New Roman;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    background:black;\n"
                                      "}")
        self.verticalLayout.addWidget(self.button_enter)

        self.button_quit = QPushButton()
        self.button_quit.setText("quit")
        self.button_quit.setStyleSheet("QPushButton{\n"
                                      "    background:#ffcc00;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px;font-size:24px;border-radius: 24px;font-family: Times New Roman;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    background:black;\n"
                                      "}")
        self.verticalLayout.addWidget(self.button_quit)

        # Bind Button Event
        self.button_enter.clicked.connect(self.button_enter_verify)
        self.button_quit.clicked.connect(
            QCoreApplication.instance().quit)

    def button_enter_verify(self):
        # Verify if the account is correct
        if self.login_id.text() != "admin":
            print("User name error!")
            return
        # Verify if the password is correct
        if self.passwd.text() != "12345":
            print("Login password error!")
            return
        # Verification passed, set QDialog object status to allow
        self.accept()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    # Set login window
    login_ui = LoginDialog()
    # Verify if the verification is successful
    if login_ui.exec_() == QDialog.Accepted:
        w = mainwindow()
        w.ui.show()
        myWin = MyMainForm()
        # 将窗口控件显示在屏幕上
        myWin.show()
        app.exec_()



    # app = QApplication(sys.argv)
    # w = mainwindow()
    # w.ui.show()
    # myWin = MyMainForm()
    # # 将窗口控件显示在屏幕上
    # myWin.show()
    # app.exec_()
