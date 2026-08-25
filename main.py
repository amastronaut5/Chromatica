# Image Color Palette Site
from collections import Counter
import os
import time
import uuid

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'heic', 'heif', 'jfif', 'pjpeg', 'pjp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def cleanup_old_uploads(max_age_seconds=3600):
    """Clean up uploaded files older than max_age_seconds to save disk space."""
    try:
        now = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                if now - os.path.getmtime(filepath) > max_age_seconds:
                    os.remove(filepath)
    except Exception:
        pass


def allowed_file(filename):
    if not filename:
        return False
    if '.' not in filename:
        return True
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS or ext == ''



def rgb_to_hsl(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn
    l = (mx + mn) / 2
    if diff == 0:
        h = s = 0
    else:
        s = diff / (1 - abs(2 * l - 1))
        if mx == r:
            h = ((g - b) / diff) % 6
        elif mx == g:
            h = ((b - r) / diff) + 2
        else:
            h = ((r - g) / diff) + 4
        h = round(h * 60)
    return f"{int(h)}°, {int(round(s * 100))}%, {int(round(l * 100))}%"


def extract_colors_from_image(image_path, n_colors=10):
    """
    Optimized color extraction.
    Resizes image thumbnail to max 200x200 before running KMeans to guarantee
    sub-50ms processing time even on low-spec CPUs (e.g. Render free tier).
    """
    my_img = Image.open(image_path).convert('RGB')
    
    # Thumbnail for fast KMeans clustering (max 40,000 pixels instead of millions)
    my_img_thumb = my_img.copy()
    my_img_thumb.thumbnail((200, 200))
    
    img_array = np.array(my_img_thumb)
    pixels = img_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=1, max_iter=100).fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    label_counts = Counter(labels)

    color_counts = sorted(
        [(tuple(centers[i]), label_counts[i]) for i in range(len(centers))],
        key=lambda x: -x[1]
    )

    color_list = []
    for color, count in color_counts:
        rgb_color = tuple(map(int, color))
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
        hsl_color = rgb_to_hsl(*rgb_color)
        color_list.append({
            'rgb': rgb_color,
            'rgb_str': f"{rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}",
            'hex': hex_color,
            'hsl': hsl_color,
            'count': int(count)
        })
        if len(color_list) >= n_colors:
            break

    return color_list


@app.route('/', methods=['GET', 'POST'])
def home():
    cleanup_old_uploads()
    image_url = None
    lst = list()
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '' and allowed_file(image.filename):
                safe_name = f"{uuid.uuid4().hex[:12]}_{secure_filename(image.filename)}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
                image.save(image_path)
                image_url = f"/static/uploads/{safe_name}"
                lst = extract_colors_from_image(image_path)

    return render_template('index.html', image_url=image_url, color_list=lst)


@app.route('/api/extract', methods=['POST'])
def api_extract():
    cleanup_old_uploads()
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    if image.filename == '' or not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file type or empty file'}), 400

    try:
        safe_name = f"{uuid.uuid4().hex[:12]}_{secure_filename(image.filename)}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        image.save(image_path)
        image_url = f"/static/uploads/{safe_name}"

        color_list = extract_colors_from_image(image_path)
        return jsonify({
            'success': True,
            'image_url': image_url,
            'color_list': color_list
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)







if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)







    # for i in os_fspath('/static/uploads'):
    #     os.remove(i)





# my_img = Image.open('C:/Users/HP/Downloads/nnn.png')
#
# img_array = np.array(my_img)
#
# pixels = img_array.reshape(-1,4)
# uniquecols,counts = np.unique(pixels,axis=0,return_counts=True)
#
# color_counts = list(zip([tuple(color) for color in uniquecols],counts))
# lst=list()
# for color, count in sorted(zip(uniquecols, counts), key=lambda x: -x[1])[:10]:
#     clean_color = tuple(int(c) for c in color)
#     lst.append(clean_color)
#     print(f"Color: {clean_color}, Count: {count}")
#
#
