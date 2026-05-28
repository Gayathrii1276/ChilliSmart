import cv2, numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.post("/analyze-image")
async def analyze_image(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Dominant color extraction
    pixels = img.reshape(-1, 3).astype(float)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3).fit(pixels)
    dominant_bgr = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    # Red mask for red mirchi detection
    red_mask1 = cv2.inRange(hsv, (0,50,50), (10,255,255))
    red_mask2 = cv2.inRange(hsv, (160,50,50), (180,255,255))
    red_ratio = (cv2.countNonZero(red_mask1 + red_mask2)) / img.size * 3
    
    # Dark spot detection (disease indicator)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    defect_ratio = cv2.countNonZero(dark_mask) / gray.size
    
    # Brightness / freshness proxy
    brightness = np.mean(hsv[:,:,2])
    saturation = np.mean(hsv[:,:,1])
    
    return {
        "red_ratio": round(red_ratio, 3),
        "defect_ratio": round(defect_ratio, 3),
        "brightness": round(float(brightness), 1),
        "saturation": round(float(saturation), 1),
        "dominant_color_hex": "#{:02x}{:02x}{:02x}".format(
            int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0]))
    }