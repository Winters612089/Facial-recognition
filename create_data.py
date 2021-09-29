import cv2
import os, glob
import numpy as np
from time import sleep

#儲存圖片函式
def saveImg(img, index):
	filename = name + '/face{:03d}.jpg'.format(index)
	cv2.imwrite(filename, img)

index, total = 1, 50  #圖片索引從1開始,取50張

name = input("請輸入姓名(請使用英文):")

if os.path.isdir(name):
	print('已有此姓名!!!')
else:
	os.mkdir(name)
	#建立臉部辨識分類器
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
		"haarcascade_frontalface_alt2.xml")
	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #開啟攝像機
	cv2.namedWindow('video', cv2.WINDOW_NORMAL)  #讀取攝像機
	while index <= total:   #照片取樣
		ret, frame = cap.read()
		frame = cv2.flip(frame, 1)  #攝像機與實際左右相反,改為正常方向
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #轉灰階
		faces = face_cascade.detectMultiScale(gray, 1.1, 3)  #偵測臉部
		for (x,y,w,h) in faces:
			#取得人臉圖形
			frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 5)  
			img = cv2.resize(gray[y: y+h, x: x+w],(300,300))  #儲存圖片格式
			saveImg(img, index)  #call函式存圖
			sleep(0.1)  #暫停0.1秒:拉長存檔時間,讓照片存檔更精確
			index += 1
			if index > total:
				print('取樣完成')
				break
		cv2.imshow('video', frame)
		cv2.waitKey(1)
	cap.release()
	cv2.destroyAllWindows()

imgs, labels, members = [], [], []  #儲存訓練圖形, 訓練標籤, 會員姓名
count = 0  #會員編號索引
dirs = os.listdir(os.getcwd())  #取得目前資料夾內所有資料夾及檔案
for d in dirs:
	if os.path.isdir(os.getcwd() + '/' + d):   #如果是資料夾才執行下列程序
		files = glob.glob(os.getcwd() + '/' + d + '/*.jpg')  #取得資料夾所有圖檔
		for filename in files:   #將會員姓名資料夾中圖片檔案加入圖片串列及標記串列
			imag = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
			imgs.append(imag)
			labels.append(count)
		members.append(d)   #將姓名加入串列
		count += 1

#建立姓名檔案, 在辨識人臉時使用
f = open('member.txt','w')
f.write(','.join(members))
f.close()

print('開始建立模型')
model = cv2.face.LBPHFaceRecognizer_create()  #建立空模型
model.train(np.asarray(imgs), np.asarray(labels))  #訓練模型
model.save('faces_LBPH.yml')  #儲存模型
print('模型建立完成!!!')