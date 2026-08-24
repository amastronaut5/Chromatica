#Image Color Palette Site
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from PIL import Image


import os
from flask import Flask,render_template,request,redirect,url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def color_delta(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



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


@app.route('/',methods=['GET','POST'])
def home():
    image_url =None
    lst = list()
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '' and allowed_file(image.filename):
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(image_path)
                image_url = f"/static/uploads/{image.filename}"

                my_img = Image.open(image_path).convert('RGB')
                img_array = np.array(my_img)
                pixels = img_array.reshape(-1,3)

                kmeans = KMeans(n_clusters=10,random_state=42).fit(pixels)
                centers = kmeans.cluster_centers_.astype(int)
                labels = kmeans.labels_
                label_counts = Counter(labels)

                color_counts = sorted([(tuple(centers[i]), label_counts[i]) for i in range(len(centers))],
                                      key=lambda x: -x[1])

                for color, count in color_counts:
                    rgb_color = tuple(map(int, color))
                    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
                    hsl_color = rgb_to_hsl(*rgb_color)
                    lst.append({
                        'rgb': rgb_color,
                        'rgb_str': f"{rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}",
                        'hex': hex_color,
                        'hsl': hsl_color,
                        'count': count
                    })

                    if len(lst) >= 10:
                        break

    return render_template('index.html', image_url=image_url, color_list=lst)






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
