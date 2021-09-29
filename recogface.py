import cv2
import time
import imutils

#載入訓練完成人臉模型
model = cv2.face.LBPHFaceRecognizer_create()
model.read('faces_LBPH.yml')
f = open('member.txt', 'r')
names = f.readline().split(',')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
	"haarcascade_frontalface_alt2.xml")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

timenow = time.time()  #取得目前時間, 倒數計時用
while(cap.isOpened()):
	count = 5 - int(time.time() - timenow)  #倒數計時
	ret, img = cap.read()
	if ret == True:
		cv2.imshow("frame", img)
		key = cv2.waitKey(100)
		if key == ord("z") or key == ord("Z") or count ==0:
			cv2.imwrite('media/pic.jpg', img)
			break

cap.release()
cv2.destroyAllWindows()

img = cv2.imread('media/pic.jpg')  #讀取剛剛拍攝的圖檔
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 3)
for (x,y,w,h) in faces:
	#取得人臉圖形
	img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 5)  
	face_img = cv2.resize(gray[y: y+h, x: x+w],(300,300))  #儲存圖片格式

	try:     #臉部差異辨識
		val = model.predict(face_img)
		if val[1] < 40:
			print('歡迎', names[val[0]], '登入~', val[1])
		else:
			print('非會員,無法登入')
	except:
		print('辨識過程產生錯誤')