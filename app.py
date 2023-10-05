from flask import Flask, render_template, request
import cv2
import numpy as np
import tempfile
import os


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'image_photo')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_crack_1(image_path, contour_threshold):
    # Đọc hình ảnh
    image = cv2.imread(image_path)

    # Sử dụng kernel để làm sắc nét hình ảnh
    kernel = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
    sharpen_image = cv2.filter2D(image, -1, kernel)

    # Áp dụng Gaussian Blur để làm mờ hình ảnh
    blurred = cv2.GaussianBlur(sharpen_image, (3, 3), 0)

    # Chuyển đổi hình ảnh thành ảnh grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Chuyển đổi ảnh grayscale thành ảnh nhị phân
    _, threshInv = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    # Tìm các đường viền trong ảnh nhị phân
    contours, _ = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #if contours:
    #    # finding the contour with maximum pixel points
    #    max_len_cnt = max([len(x) for x in contours])
    #    if max_len_cnt < 100:
    #        return True
    #    else:
    #        return False

    #else:
    #    return True
    # Tạo một bản sao của hình ảnh để vẽ đường viền lên đó
    output_image = image.copy()

    # Vẽ các đường viền lên bản sao của hình ảnh
    for contour in contours:
        if len(contour) > contour_threshold:
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

    # Lưu hình ảnh vào tệp tạm thời
    temp_image_path = tempfile.mktemp(suffix='.jpg', dir=app.config['UPLOAD_FOLDER'])
    cv2.imwrite(temp_image_path, output_image)

    return temp_image_path


def detect_crack_check(image_name, contour_threshold):
    #read image
    image = cv2.imread(image_name)
    #sharpen the image to fine-tune the edges using the kernel
    kernel = np.array([[-1,-1,-1], [-1,11,-1], [-1,-1,-1]])
    sharpen_image = cv2.filter2D(image, -1, kernel)
    #filter out small edges by using blur method
    blurred = cv2.GaussianBlur(sharpen_image, (3, 3), 0)
    #convert image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #convert grayscale image to binary image
    (T, threshInv) = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    #detect contours in binary image
    contours, h = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    if contours:
        #finding the contour with maximum pixel points
        max_len_cnt = max([len(x) for x in contours])
        if max_len_cnt < 100:
            return True
        else:
            return False

    else:
        return True

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Lưu file ảnh tải lên vào thư mục uploads
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = tempfile.mktemp(suffix='.jpg', dir=app.config['UPLOAD_FOLDER'])
            uploaded_file.save(file_path)

            # Kiểm tra vết nứt và trả về kết quả
            contour_threshold = 100  # Ngưỡng đường viền
            result_image_path = detect_crack_1(file_path, contour_threshold)
            is_crack_detected = detect_crack_check(file_path, contour_threshold)
            return render_template('upload.html', result_image=result_image_path,is_crack_detected=is_crack_detected)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)