import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import restoration
import pandas as pd
from skimage.measure import label, regionprops
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.signal import convolve2d
from scipy.signal import wiener


class DIPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dijital Görüntü İşleme")
        self.root.geometry("900x650")

        self.header = tk.Label(root, text="Dijital Görüntü İşleme\nNevra Nur Doğan\n221229058",
                               font=("Arial", 14, "bold"))
        self.header.pack(pady=10)

        self.menu_bar = tk.Menu(root)
        root.config(menu=self.menu_bar)

        self.home_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Ödevler", menu=self.home_menu)
        self.home_menu.add_command(label="Ödev 1: Temel İşlevsellik", command=lambda: self.show_homework(1))
        self.home_menu.add_command(label="Ödev 2: Filtre Uygulama", command=lambda: self.show_homework(2))
        self.home_menu.add_command(label="Ödev 3: Histogram Analizi", command=lambda: self.show_homework(3))
        self.home_menu.add_command(label="Ödev 4: Final Ödevi", command=lambda: self.show_homework(4))


        self.homework_frame = tk.Frame(root)
        self.homework_frame.pack(pady=20)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        self.image_path = None
        self.loaded_image = None
        self.explanation_visible = False

    def toggle_explanation(self, text, button):
        if self.explanation_visible:
            self.explanation_label.destroy()
            button.config(text="Açıklamayı Göster")
        else:
            self.explanation_label = tk.Label(self.homework_frame, text=text, font=("Arial", 11), wraplength=600,
                                              justify="left")
            self.explanation_label.pack()
            button.config(text="Açıklamayı Gizle")
        self.explanation_visible = not self.explanation_visible

    def show_homework(self, hw_num):
        for widget in self.homework_frame.winfo_children():
            widget.destroy()

        if hw_num == 1:
            text = "Amaç: Görüntü üzerinde temel işlemleri yapmak\n\n"
            text += "- Görüntü Ekle: Kullanıcıdan resim seçme.\n"
            text += "- Gri Tonlama: Görüntüyü siyah-beyaz yapma.\n"
            text += "- Negatif Dönüştürme: Renkleri ters çevirme.\n"
            text += "- Eşikleme: Belirli bir parlaklık değerine göre siyah-beyaz dönüşümü."

            button = tk.Button(self.homework_frame, text="Açıklamayı Göster",
                               command=lambda: self.toggle_explanation(text, button))
            button.pack()

            tk.Button(self.homework_frame, text="Görüntü Ekle", command=self.open_image).pack(pady=5)
            tk.Button(self.homework_frame, text="Gri Tonlama", command=self.apply_grayscale).pack(pady=5)
            tk.Button(self.homework_frame, text="Negatif Dönüştürme", command=self.apply_negative).pack(pady=5)
            tk.Button(self.homework_frame, text="Eşikleme", command=self.apply_thresholding).pack(pady=5)

        elif hw_num == 2:
            text = "Amaç: Görüntüye farklı filtreler uygulamak\n\n"
            text += "- Bulanıklık (Blur): Görüntüyü bulanık hale getirme.\n"
            text += "- Keskinleştirme (Sharpen): Detayları daha belirgin hale getirme.\n"
            text += "- Kenar Bulma (Edge Detection): Görüntüdeki kenarları tespit etme.\n"
            text += "- Kontur Çıkarma (Contour): Nesnelerin dış hatlarını belirginleştirme."

            button = tk.Button(self.homework_frame, text="Açıklamayı Göster",
                               command=lambda: self.toggle_explanation(text, button))
            button.pack()

            tk.Button(self.homework_frame, text="Görüntü Ekle", command=self.open_image).pack(pady=5)
            tk.Button(self.homework_frame, text="Bulanıklık (Blur)",
                      command=lambda: self.apply_filter(ImageFilter.BLUR)).pack(pady=5)
            tk.Button(self.homework_frame, text="Keskinleştirme",
                      command=lambda: self.apply_filter(ImageFilter.SHARPEN)).pack(pady=5)
            tk.Button(self.homework_frame, text="Kenar Bulma",
                      command=lambda: self.apply_filter(ImageFilter.FIND_EDGES)).pack(pady=5)
            tk.Button(self.homework_frame, text="Kontur Çıkarma",
                      command=lambda: self.apply_filter(ImageFilter.CONTOUR)).pack(pady=5)

        elif hw_num == 3:
            text = "Amaç: Görüntünün renk dağılımını analiz etmek\n\n"
            text += "- RGB Kanalı Ayrımı: Kırmızı, yeşil ve mavi renk kanallarını ayrı ayrı gösterme."

            button = tk.Button(self.homework_frame, text="Açıklamayı Göster",
                               command=lambda: self.toggle_explanation(text, button))
            button.pack()

            tk.Button(self.homework_frame, text="Görüntü Ekle", command=self.open_image).pack(pady=5)
            tk.Button(self.homework_frame, text="Histogram Görüntüle", command=self.view_histogram).pack(pady=5)
            tk.Button(self.homework_frame, text="RGB Kanalı Ayrımı", command=self.apply_rgb).pack(pady=5)
            
            
        elif hw_num == 4:
            text = "Amaç: Gelişmiş görüntü işleme görevlerini gerçekleştirmek\n\n"
            text += "- S-Curve ile Kontrast Artırımı\n"
            text += "- Hough Dönüşümü ile Kenar/Özellik Tespiti\n"
            text += "- Deblurring ile Görüntü Netleştirme\n"
            text += "- Nesne Sayımı ve Özellik Çıkarımı"

            button = tk.Button(self.homework_frame, text="Açıklamayı Göster",
                            command=lambda: self.toggle_explanation(text, button))
            button.pack()

            tk.Button(self.homework_frame, text="Görüntü Ekle", command=self.open_image).pack(pady=5)
            tk.Button(self.homework_frame, text="1.a-)S-Curve Kontrast Artırımı", command=self.apply_standard_sigmoid).pack(pady=5)
            tk.Button(self.homework_frame, text="1.b-)Shifted Sigmoid", command=self.apply_shifted_sigmoid).pack(pady=5)
            tk.Button(self.homework_frame, text="1.c-)Steep Sigmoid", command=self.apply_steep_sigmoid).pack(pady=5)
            tk.Button(self.homework_frame, text="1.d-)Özel Sigmoid Fonksiyon", command=self.apply_custom_function).pack(pady=5)
            tk.Button(self.homework_frame, text="2.a-)Yoldaki Çizgileri Tespit Et", command=self.detect_road_lines).pack(pady=5)
            tk.Button(self.homework_frame, text="2.b-)Göz Tespiti (Hough ile)", command=self.detect_eyes).pack(pady=5)
            tk.Button(self.homework_frame, text="3-)Deblurring Uygula", command=self.apply_motion_deblurring).pack(pady=5)
            tk.Button(self.homework_frame, text="4-)Yeşil Nesne Analizi", command=self.object_count_and_features).pack(pady=5)



    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            self.loaded_image = Image.open(self.image_path)
            self.display_image(self.loaded_image)

    def display_image(self, image):
        img = image.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def apply_filter(self, filter_type):
        if self.loaded_image:
            self.display_image(self.loaded_image.filter(filter_type))

    def apply_grayscale(self):
        if self.loaded_image:
            self.display_image(self.loaded_image.convert('L'))

    def apply_negative(self):
        if self.loaded_image:
            self.display_image(ImageOps.invert(self.loaded_image.convert('RGB')))

    def apply_thresholding(self):
        if self.loaded_image:
            gray_image = self.loaded_image.convert('L')
            np_image = np.array(gray_image)
            thresholded_image = Image.fromarray(np.where(np_image > 128, 255, 0).astype('uint8'))
            self.display_image(thresholded_image)

    def view_histogram(self):
        if self.loaded_image:
            plt.hist(np.array(self.loaded_image.convert('L')).ravel(), bins=256, color='black')
            plt.show()

    def apply_rgb(self):
        if self.loaded_image:
            r, g, b = self.loaded_image.split()
            plt.figure(figsize=(10, 3))
            plt.subplot(131), plt.imshow(r, cmap='Reds'), plt.title("Red")
            plt.subplot(132), plt.imshow(g, cmap='Greens'), plt.title("Green")
            plt.subplot(133), plt.imshow(b, cmap='Blues'), plt.title("Blue")
            plt.show()
            
    def apply_standard_sigmoid(self):
        if self.loaded_image:
            img_gray = self.loaded_image.convert('L')
            img_array = np.asarray(img_gray).astype(np.float32)
            normalized = img_array / 255.0
            sigmoid = 1 / (1 + np.exp(-10 * (normalized - 0.5)))
            output = (sigmoid * 255).astype(np.uint8)
            self.display_image(Image.fromarray(output))
            
    def apply_shifted_sigmoid(self):
        if self.loaded_image:
            img_gray = self.loaded_image.convert('L')
            img_array = np.asarray(img_gray).astype(np.float32)
            normalized = img_array / 255.0
            x0 = 0.3 
            result = 1 / (1 + np.exp(-10 * (normalized - x0)))
            output = (result * 255).astype(np.uint8)
            self.display_image(Image.fromarray(output))
            
    def apply_steep_sigmoid(self):
        if self.loaded_image:
            img_gray = self.loaded_image.convert('L')
            img_array = np.asarray(img_gray).astype(np.float32)
            normalized = img_array / 255.0
            result = 1 / (1 + np.exp(-20 * (normalized - 0.5)))  
            output = (result * 255).astype(np.uint8)
            self.display_image(Image.fromarray(output))
            
    def apply_custom_function(self):
        if self.loaded_image:
            img_gray = self.loaded_image.convert('L')
            img_array = np.asarray(img_gray).astype(np.float32)
            normalized = img_array / 255.0
            result = np.sqrt(normalized) ** 1.2 
            output = (result * 255).astype(np.uint8)
            self.display_image(Image.fromarray(output))

    def detect_road_lines(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            if img is None:
                print("Görüntü okunamadı. Dosya yolunu kontrol edin.")
                return

            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    
            mask = np.zeros_like(edges)
            polygon = np.array([[
                (0, height),
                (0, height * 0.6),
                (width, height * 0.6),
                (width, height)
            ]], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)

            lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

            if lines is not None:
                left_lines = []
                right_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 == x1:
                        continue  
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < 0.5: 
                        continue
                    if slope < 0:
                        left_lines.append(line[0])
                    else:
                        right_lines.append(line[0])

            def average_line(lines):
                if len(lines) == 0:
                    return None
                x_coords = []
                y_coords = []
                for x1, y1, x2, y2 in lines:
                    x_coords.extend([x1, x2])
                    y_coords.extend([y1, y2])
                poly = np.polyfit(x_coords, y_coords, 1) 
                m, b = poly

                y1 = height
                y2 = int(height * 0.6)
                x1 = int((y1 - b) / m)
                x2 = int((y2 - b) / m)
                return (x1, y1, x2, y2)

            left_avg = average_line(left_lines)
            right_avg = average_line(right_lines)

            if left_avg is not None:
                cv2.line(img, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 0), 5)
            if right_avg is not None:
                cv2.line(img, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 0), 5)

        cv2.imshow("Yol Çizgileri", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    def detect_eyes(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            if img is None:
                print("Görüntü okunamadı. Dosya yolunu kontrol edin.")
                return

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            cv2.imshow("Göz Tespiti", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def apply_motion_deblurring(self):
        if self.image_path:
      
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float64) / 255.0

           
            def motion_blur_psf(length, angle):
                size = length if length % 2 == 1 else length + 1
                psf = np.zeros((size, size))
                center = size // 2
                angle = np.deg2rad(angle)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                for i in range(length):
                    x = int(center + (i - length//2) * cos_a)
                    y = int(center + (i - length//2) * sin_a)
                    if 0 <= x < size and 0 <= y < size:
                        psf[y, x] = 1
                psf /= psf.sum()
                return psf

            psf = motion_blur_psf(length=15, angle=0)

           
            def richardson_lucy(image, psf, iterations=30):
                img_deconv = np.full(image.shape, 0.5)
                psf_mirror = psf[::-1, ::-1]
                for i in range(iterations):
                    conv = convolve2d(img_deconv, psf, 'same')
                    relative_blur = image / (conv + 1e-7)
                    img_deconv *= convolve2d(relative_blur, psf_mirror, 'same')
                return img_deconv

          
            deblurred = richardson_lucy(img, psf, iterations=40)

            
            deblurred = np.clip(deblurred, 0, 1)
            deblurred = (deblurred * 255).astype(np.uint8)

            
            img_pil = Image.fromarray(deblurred)
            self.display_image(img_pil)
    

    def object_count_and_features(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            
            lower_green = np.array([30, 50, 20])
            upper_green = np.array([90, 255, 150])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            data = []

            for i in range(1, num_labels):  
                x, y, w, h, area = stats[i]
                cx, cy = centroids[i]

                roi = mask[y:y + h, x:x + w]

                diagonal = int(np.sqrt(w ** 2 + h ** 2))
                energy = np.sum((roi / 255.0) ** 2)
                hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
                p = hist / np.sum(hist)
                entropy = -np.sum(p * np.log2(p + 1e-9))
                mean_val = int(np.mean(roi))
                median_val = int(np.median(roi))

                data.append({
                    "No": i,
                    "Center": f"{int(cx)},{int(cy)}",
                    "Length": f"{h} px",
                    "Width": f"{w} px",
                    "Diagonal": f"{diagonal} px",
                    "Energy": round(energy, 3),
                    "Entropy": round(entropy, 2),
                    "Mean": mean_val,
                    "Median": median_val
                })

             
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

          
            df = pd.DataFrame(data)
            df.to_excel("nesne_ozellikleri.xlsx", index=False)
            print("Excel dosyasına yazıldı: nesne_ozellikleri.xlsx")

            cv2.imshow("Nesne Tespiti", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.show_table(df)

    def show_table(self, df):
       
        for widget in self.homework_frame.winfo_children():
            if isinstance(widget, ttk.Treeview):
                widget.destroy()

        tree = ttk.Treeview(self.homework_frame, columns=list(df.columns), show="headings")
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor="center")

        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(pady=10)


root = tk.Tk()
app = DIPApp(root)
root.mainloop()

