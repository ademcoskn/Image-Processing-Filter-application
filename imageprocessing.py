from fileinput import filename
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QHBoxLayout
import sys
from PyQt5.QtGui import QPixmap
import cv2
from cv2 import COLOR_BGR2GRAY
from cv2 import GaussianBlur
import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtCore


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.title = "Görüntü İsleme"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300
        self.InitUI()

    def InitUI(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        hbox5 = QHBoxLayout()
        hbox6 = QHBoxLayout()
        hbox7 = QHBoxLayout()
        hbox8 = QHBoxLayout()
        hbox9 = QHBoxLayout()
        self.btn1 = QPushButton("Görüntüyü Seç")            #
        self.btn2 = QPushButton("Griye Çevir")
        self.btn3 = QPushButton("Pencere - Seviye")
        self.btn4 = QPushButton("Negatif")
        self.btn5 = QPushButton("Medyan")
        self.btn6 = QPushButton("Mean")
        self.btn7 = QPushButton("Aşınma")
        self.btn8 = QPushButton("Resmi Kirp")
        self.btn9 = QPushButton("Yazi Ekle")
        self.btn10 = QPushButton("Resimi Çevir")
        self.btn11 = QPushButton("Canny Filtresi")
        self.btn12 = QPushButton("Gaussian Filtresi")
        self.btn13 = QPushButton("Piksel")
        self.btn14 = QPushButton("Maskeleme Filtresi")
        self.btn15 = QPushButton("Genleşme")
        self.btn16 = QPushButton("Gradient")
        self.btn17 = QPushButton("Tophat")
        self.btn18 = QPushButton("Blakhat")
        self.btn19 = QPushButton("Yayma")
        self.btn20 = QPushButton("Açınım")
        self.btn21 = QPushButton("Kapanım")
        self.btn22 = QPushButton("Histogram Grafik")   #
        self.btn23 = QPushButton("Laplace")
        self.btn24 = QPushButton("Sobel")
        self.btn25 = QPushButton("Eşikleme")
        self.btn26 = QPushButton("Histogram Eşitleme") #
        self.btn27 = QPushButton("Kontrast Germe")     #
        self.btn28 = QPushButton("Pencere - Genişlik") #
        self.btn29 = QPushButton("Zincir Kodlama")
        self.btn30 = QPushButton("Görüntü Eşleştirme")
        self.btn31 = QPushButton("Yansıtma")
        self.btn32 = QPushButton("Daraltma")
        self.btn33 = QPushButton("Genişletme")
        self.btn34 = QPushButton("Köşe Noktası")
        self.btn35 = QPushButton("Uç Nokta")
        self.btn36 = QPushButton("İskelet")
        self.btn37 = QPushButton("Bölge Genişletme")
        self.btn38 = QPushButton("Watershed")
        self.btn39 = QPushButton("Prewit")
        self.btn40 = QPushButton("Roberts")
        self.btn41 = QPushButton("Keskinleştirme")
        self.btn42 = QPushButton("HSV")


        # LAYOUTLARA BUTONLARI EKLEME
        vbox.addWidget(self.btn1)
        hbox1.addWidget(self.btn2)
        hbox1.addWidget(self.btn3)
        hbox1.addWidget(self.btn4)
        hbox1.addWidget(self.btn5)
        hbox2.addWidget(self.btn6)
        hbox2.addWidget(self.btn7)
        hbox2.addWidget(self.btn8)
        hbox2.addWidget(self.btn9)
        hbox3.addWidget(self.btn10)
        hbox3.addWidget(self.btn11)
        hbox3.addWidget(self.btn12)
        hbox3.addWidget(self.btn13)
        hbox4.addWidget(self.btn14)
        hbox4.addWidget(self.btn15)
        hbox4.addWidget(self.btn16)
        hbox4.addWidget(self.btn17)
        hbox5.addWidget(self.btn18)
        hbox5.addWidget(self.btn19)
        hbox5.addWidget(self.btn20)
        hbox5.addWidget(self.btn21)
        hbox6.addWidget(self.btn22)
        hbox6.addWidget(self.btn23)
        hbox6.addWidget(self.btn24)
        hbox6.addWidget(self.btn25)
        hbox7.addWidget(self.btn26)
        hbox7.addWidget(self.btn27)
        hbox7.addWidget(self.btn28)
        hbox7.addWidget(self.btn29)
        hbox8.addWidget(self.btn30)
        hbox8.addWidget(self.btn31)
        hbox8.addWidget(self.btn32)
        hbox8.addWidget(self.btn33)
        hbox9.addWidget(self.btn34)
        hbox9.addWidget(self.btn35)
        hbox9.addWidget(self.btn36)
        hbox9.addWidget(self.btn37)
        hbox2.addWidget(self.btn38)
        hbox5.addWidget(self.btn39)
        hbox9.addWidget(self.btn40)
        hbox3.addWidget(self.btn41)
        hbox6.addWidget(self.btn42)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        vbox.addLayout(hbox6)
        vbox.addLayout(hbox7)
        vbox.addLayout(hbox8)
        vbox.addLayout(hbox9)
        self.label = QLabel("")
        self.label2 = QLabel("193302103 - Adem COŞKUN")
        self.label2.setStyleSheet("color : brown")

        vbox.addWidget(self.label)
        vbox.addStretch()
        vbox.addWidget(self.label2)

        self.setLayout(vbox)

        # FONKSİYONLARI ÇAĞIRMA

        self.btn1.clicked.connect(self.getImage)
        self.btn2.clicked.connect(self.griyeCevir)
        self.btn3.clicked.connect(self.pencereseviye)
        self.btn4.clicked.connect(self.negatif)
        self.btn5.clicked.connect(self.medyan)
        self.btn6.clicked.connect(self.mean)
        self.btn7.clicked.connect(self.asinma)
        self.btn8.clicked.connect(self.resmikirp)
        self.btn9.clicked.connect(self.resmeyaziyazdirma)
        self.btn10.clicked.connect(self.goruntucevirme)
        self.btn11.clicked.connect(self.canny)
        self.btn12.clicked.connect(self.gaus)
        self.btn13.clicked.connect(self.pixel)
        self.btn14.clicked.connect(self.maskeleme)
        self.btn15.clicked.connect(self.genlesme)
        self.btn16.clicked.connect(self.gradient)
        self.btn17.clicked.connect(self.tophat)
        self.btn18.clicked.connect(self.blackhat)
        self.btn19.clicked.connect(self.yayma)
        self.btn20.clicked.connect(self.acma)
        self.btn21.clicked.connect(self.kapama)
        self.btn22.clicked.connect(self.histogram_grafik)
        self.btn23.clicked.connect(self.laplace)
        self.btn24.clicked.connect(self.sobel)
        self.btn25.clicked.connect(self.esikleme)
        self.btn26.clicked.connect(self.histogram)
        self.btn27.clicked.connect(self.kontrastgerme)
        self.btn28.clicked.connect(self.penceregenislik)
        self.btn29.clicked.connect(self.chaincoding)
        self.btn30.clicked.connect(self.imagematching)
        self.btn31.clicked.connect(self.reflection)
        self.btn32.clicked.connect(self.daraltma)
        self.btn33.clicked.connect(self.genisletme)
        self.btn34.clicked.connect(self.koseNoktaBelirleme)
        self.btn35.clicked.connect(self.noktaVeKoseNoktasi)
        self.btn36.clicked.connect(self.iskeleteDonusturme)
        self.btn37.clicked.connect(self.regionGrowing)
        self.btn38.clicked.connect(self.watershedDonusum)
        self.btn39.clicked.connect(self.prewit)
        self.btn40.clicked.connect(self.roberts)
        self.btn41.clicked.connect(self.sharpening)
        self.btn42.clicked.connect(self.hsv)
        self.show()

    # FONKSİYONLARI TANIMLAMA
    def getImage(self):
        global imagePath
        global fname
        global pixmap
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c://', "Image files (*.jpg *.gif *.jpeg *.png *.tif)")
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.label.setAlignment(QtCore.Qt.AlignCenter)

    def griyeCevir(self):
        resim = cv2.imread("{}".format(imagePath), 0)
        cv2.imshow("Gri Resim", resim)

    def pencereseviye(self):
        value=200
        resim = cv2.imread("{}".format(imagePath))
        hsv = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
        
        for x in range(0, len(hsv)):
            for y in range(0, len(hsv[0])):
                hsv[x,y][2] += value
                
        resim=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Window Level", resim)

    def negatif(self):
        resim = cv2.imread("{}".format(imagePath))
        negatif = cv2.bitwise_not(resim)
        cv2.imshow("negatif resim", negatif)

    def medyan(self):
        resim = cv2.imread("{}".format(imagePath))
        medyan = cv2.medianBlur(resim, 5)
        cv2.imshow("medyan filtresi", medyan)

    def mean(self, plt=None):
        resim = cv2.imread("{}".format(imagePath))
        ortalama = cv2.blur(resim, (3, 3))
        res = np.hstack((resim, ortalama))
        cv2.imshow('res.jpg', res)

    def asinma(self):
        resim = cv2.imread("{}".format(imagePath),0)
        kernel = np.ones((5,5), np.uint8)
        
        img_erosion=cv2.erode(resim, kernel, iterations=1)
        
        cv2.imshow('Genlesme', img_erosion)

    def resmikirp(self):
        resim = cv2.imread("{}".format(imagePath))
        kirpma = resim[60:360, 20:360]
        cv2.imshow("kirpilmis resim", kirpma)

    def resmeyaziyazdirma(self):
        resim = cv2.imread("{}".format(imagePath))
        renk = (0, 0, 255)
        resim = cv2.putText(resim, "deneme", (50, 50), cv2.FONT_ITALIC, 1, renk, 2)
        cv2.imshow("Resme Yazi Yazdirma", resim)

    def goruntucevirme(self):
        img = cv2.imread("{}".format(imagePath), 0)
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
        dst = cv2.warpAffine(img, M, (cols,rows))
        cv2.imshow("90 derece dondurulmus goruntu", dst)

    def canny(self):
        img = cv2.imread("{}".format(imagePath), 0)
        Canny = cv2.Canny(img, 100, 200)
        
        titles = ['image', 'canny']
        images = [img, Canny]
        for i in range(2):
            plt.subplot(1, 2, i+1), plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    def gaus(self):
        resim = cv2.imread("{}".format(imagePath))
        bulanık = cv2.GaussianBlur(resim, (5, 5), 5)
        cv2.imshow("Gaussian", bulanık)

    def pixel(self):
        resim = cv2.imread("{}".format(imagePath))
        az = np.array([25, 50, 25])
        cok = np.array([255, 255, 255])
        inrange = cv2.inRange(resim, az, cok)
        cv2.imshow("belli pixel ", inrange)

    def maskeleme(self):
        resim = cv2.imread("{}".format(imagePath))
        az = np.array([25, 50, 25])
        cok = np.array([255, 255, 255])
        inrange = cv2.inRange(resim, az, cok)
        sonuc = cv2.bitwise_and(resim, resim, mask=inrange)
        cv2.imshow("maskeleme ", sonuc)

    def genlesme(self):
        resim = cv2.imread("{}".format(imagePath),0)
        kernel = np.ones((5,5), np.uint8)
        
        img_dilation=cv2.dilate(resim, kernel, iterations=1)
        
        cv2.imshow('Genlesme', img_dilation)

    def gradient(self):
        resim = cv2.imread("{}".format(imagePath))
        cekirdek = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(resim, cv2.MORPH_GRADIENT, cekirdek)
        cv2.imshow("gradient görüntü", gradient)

    def tophat(self):
        resim = cv2.imread("{}".format(imagePath))
        cekirdek = np.ones((7, 7), np.uint8)
        gradient = cv2.morphologyEx(resim, cv2.MORPH_TOPHAT, cekirdek)
        cv2.imshow("top hat", gradient)

    def blackhat(self):
        resim = cv2.imread("{}".format(imagePath))
        cekirdek = np.ones((7, 7), np.uint8)
        gradient = cv2.morphologyEx(resim, cv2.MORPH_BLACKHAT, cekirdek)
        cv2.imshow("black hat", gradient)

    def yayma(self):
        resim = cv2.imread("{}".format(imagePath))
        cekirdek = np.ones((5, 5), np.uint8)
        gradient = cv2.dilate(resim, cekirdek, iterations=2)
        cv2.imshow("yayma", gradient)

    def acma(self):
        resim = cv2.imread("{}".format(imagePath),0)
        cekirdek = np.ones((5, 5), np.uint8)
        acma = cv2.morphologyEx(resim, cv2.MORPH_OPEN, cekirdek)
        cv2.imshow("Acilmis resim", acma)

    def kapama(self):
        resim = cv2.imread("{}".format(imagePath),0)
        cekirdek = np.ones((5, 5), np.uint8)
        kapama = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, cekirdek)
        cv2.imshow("Acilmis resim", kapama)

    def histogram_grafik(self):  # HATA
        resim = cv2.imread("{}".format(imagePath))
        b, g, r = cv2.split(resim)
        plt.hist(b.ravel(), 256, [0, 256])
        plt.hist(g.ravel(), 256, [0, 256])
        plt.hist(r.ravel(), 256, [0, 256])
        hist = cv2.calcHist([resim], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.show()

    def laplace(self):
        resim = cv2.imread("{}".format(imagePath))
        laplace = cv2.Laplacian(resim, cv2.CV_64F)
        cv2.imshow("laplace filtresi", laplace)

    def sobel(self):
        resim = cv2.imread("{}".format(imagePath), 0)
        sobel = cv2.Sobel(resim, cv2.CV_8U, 1, 0, ksize=5)
        cv2.imshow("sobel filtresi", sobel)

    def esikleme(self):
        resim = cv2.imread("{}".format(imagePath), 0)
        ret, thresh = cv2.threshold(resim, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow("Esikleme", thresh)

    def histogram(self):
        resim = cv2.imread("{}".format(imagePath), 0)
        #griresim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        histogram = cv2.equalizeHist(resim)
        cv2.imshow('Histogram Esitleme', histogram)
        
    def kontrastgerme(self):
        resim = cv2.imread("{}".format(imagePath), cv2.IMREAD_COLOR)
        norm_img1 = cv2.normalize(resim, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = cv2.normalize(resim, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        norm_img1=(255*norm_img1).astype(np.uint8)
        norm_img2=np.clip(norm_img2, 0, 1)
        norm_img2=(255*norm_img2).astype(np.uint8)
        
        cv2.imshow("Kontrast germe", norm_img2)
        
    def penceregenislik(self):
        img=cv2.imread("{}".format(imagePath))
        alpha=2
        beta=50
        result=cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        cv2.imshow("Window", result)

    def chaincoding(self):
        img = cv2.imread("{}".format(imagePath),0)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        num_labels, labels_im = cv2.connectedComponents(img)
        
        def imshow_components(labels):
            label_hue = np.uint8(179*labels/np.max(labels))
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            
            labeled_img[label_hue==0] = 0
            
            cv2.imshow("Zincir Kodlama", labeled_img)
        
        imshow_components(labels_im)

    def imagematching(self):
        img = cv2.imread("{}".format(imagePath),0)
        img2 = img.copy()
        template = cv2.imread("{}".format(imagePath),0)
        w, h = template.shape[::-1]
        
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                   'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, 255, 2)
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()
            
    def reflection(self):
        img = cv2.imread("{}".format(imagePath))
        rows, cols, ch = img.shape
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (cols,rows))
        cv2.imshow("Yansitma", dst)

    def daraltma(self):
        img = cv2.imread("{}".format(imagePath),0)
        rows, cols = img.shape
        M = np.float32([[1,0,100],[0,1,50]])
        dst = cv2.warpAffine(img, M, (cols,rows))
        cv2.imshow('Daraltma', dst)
        
    def genisletme(self):
        img = cv2.imread("{}".format(imagePath))
        rows, cols, ch = img.shape
        pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
        pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (300,300))
        cv2.imshow('Genisletme', dst)
        
    def koseNoktaBelirleme(self):
        img = cv2.imread("{}".format(imagePath))
        
        gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gri = np.float32(gri)
        corner = cv2.goodFeaturesToTrack(gri, 50, 0.01, 100)
        corner = np.int0(corner)
        
        for c in corner:
            x,y=c.ravel()
            cv2.circle(img, (x,y), 5, (255,0,0), -1)
        cv2.imshow("Kose Nokta Belirleme", img)
        
    def noktaVeKoseNoktasi(self):
        image = cv2.imread("{}".format(imagePath))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,5))
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,150))
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        joints = cv2.bitwise_and(horizontal, vertical)
        
        cnts = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts [1]
        for c in cnts:
            M= cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 15, (36,255,12), -1)
            
        cv2.imshow('Horizontal', horizontal)
        cv2.imshow('Vertical', vertical)
        cv2.imshow('Joints', joints)
        cv2.imshow('Image', image)
        
    def iskeleteDonusturme(self):
        img = cv2.imread("{}".format(imagePath), 0)
        ret,img = cv2.threshold(img, 127, 255, 0)
        
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        while True:
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img)==0:
                break
            
        cv2.imshow("Iskelete Donusturme", skel)
      
    def regionGrowing(self):
        def on_mouse(event, x, y, flags, params):
            if event == cv2.CV_EVENT_LBUTTONDOWN:
                print ('Start Mouse Position: ' + str(x) + ', ' + str(y))
                s_box = x, y
                boxes.append(s_box)
                
        def region_growing(img, seed):
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            region_threshold = 0.2
            region_size = 1
            intensity_difference = 0
            neighbor_point_list = []
            neighbor_intensity_list = []
            region_mean = img[seed]
            height, width = img.shape
            image_size = height * width
            segmented_img = np.zeros((height, width, 1), np.uint8)
            
            while(intensity_difference < region_threshold) & (region_size < image_size):
                for i in range(4):
                    x_new = seed[0] + neighbors[i][0]
                    y_new = seed[1] + neighbors[i][1]
                    check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)
                    if check_inside:
                        if segmented_img[x_new, y_new] == 0:
                            neighbor_point_list.append(img[x_new, y_new])
                            neighbor_intensity_list.append(img[x_new, y_new])
                            segmented_img[x_new, y_new] = 255
                distance = abs(neighbor_intensity_list - region_mean)
                pixel_distance = min(distance)
                index = np.where(distance == pixel_distance)[0][0]
                segmented_img[seed[0], seed[1]] = 255
                region_size += 1
                region_mean = (region_mean - region_size + neighbor_intensity_list[index])/(region_size + 1)
                seed = neighbor_point_list[index]
                neighbor_intensity_list[index] = neighbor_intensity_list[-1]
                neighbor_point_list[index] = neighbor_point_list[-1]
            return segmented_img
        
        if __name__ == '__main__':
            boxes = []
            filename = 'image.jpg'
            img = cv2.imread("{}".format(imagePath), 0)
            resized = cv2.resize(img, (256, 256))
            cv2.namedWindow('input')
            cv2.setMouseCallback('input', on_mouse, 0)
            cv2.imshow('input', resized)
            cv2.waitKey()
            print("Starting region growing based on last click")
            seed = boxes[-1]
            cv2.imshow('input', region_growing(resized, seed))
            print("Done. Showing output now")
            
    def watershedDonusum(self):
        img = cv2.imread("{}".format(imagePath), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('Watershed', thresh)
        
    def prewit(self):
        img = cv2.imread("{}".format(imagePath), 0)
        prewitx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewity = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        img_prewitx = cv2.filter2D(img, -1, prewitx)
        img_prewity = cv2.filter2D(img, -1, prewity)
        prewit = cv2.add(img_prewitx, img_prewity)
        cv2.imshow("Prewit", prewit)
    
    def roberts(self):
        img = cv2.imread("{}".format(imagePath), 0)
        robertx = np.array([[1,0],[0,-1]])
        roberty = np.array([[0,1],[-1,0]])
        img_robertx = cv2.filter2D(img, -1, robertx)
        img_roberty = cv2.filter2D(img, -1, roberty)
        robert = cv2.add(img_robertx, img_roberty)
        cv2.imshow("Robert", robert)
        
    def sharpening(self):
        img = cv2.imread("{}".format(imagePath))
        gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)
        sharpened = cv2.addWeighted(img, 7.5, gaussian_blur, -6.5, 0)
        cv2.imshow("Keskinlestirme", sharpened)
    
    def hsv(self):
        resim = cv2.imread("{}".format(imagePath))
        hsv = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV görüntÜ", hsv)

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
