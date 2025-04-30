import os
import base64
import json
import cv2
import dash
from dash import dcc, html, Input, Output, State
import torch


app = dash.Dash(__name__)

def detect_objects(img_dir, output_dir, info_file, conf=0.75, threshold=1, target_type="person"):
    # Target corrspond to label in dataset
    target_map = {
        "person": ["person"],
        "pets": ["cat", "dog"],
        "vehicles": ["car", "truck", "bus", "motorcycle"],
        "airplane" : ["airplane"]
    }

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = conf  # model confident

    os.makedirs(output_dir, exist_ok=True)
    info_list = []
    info_list_2 = []

    for file in os.listdir(img_dir):
        path = os.path.join(img_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue

        results = model(img, size=640)
        results.render()
        out_img = results.ims[0]

        # get detect result in dataframe style
        detections = results.pandas().xyxy[0]

        target_labels = target_map.get(target_type, ["person"])
        count = detections[detections['name'].isin(target_labels)].shape[0]

        # threshold for present on web page
        if count >= threshold:
            out_path = os.path.join(output_dir, file)
            cv2.imwrite(out_path, out_img)
            info_list.append({
                "file_name": file,
                "num_persons": int(count)
            })

        info_list_2.append({
            "file_name": file,
            "num_persons": int(count)
        })

    with open(info_file, "w") as f:
        json.dump(info_list, f, indent=2)

    return info_list, info_list_2


def extract_frames_from_video(video_path, output_dir, frame_interval=10):
    # Get a frame and skip several frame, decided by frame_interval
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot Open: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Total frame saved: {saved_count}")


def encode_image(image_path):
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return "data:image/jpeg;base64," + encoded


# --------------------
# Dash Layout (pass parameters and run the GUI logic)
# --------------------
app.layout = html.Div([
    html.H2("YOLOv5 Image Object Detection Dashboard"),

    # parameters in web page, whill pass to the run_detection then detect_objects
    html.Div([
        
        html.Div([
            html.Label("Target Type:"),
            dcc.Dropdown(
                id="target-type",
                options=[
                    {"label": "Person", "value": "person"},
                    {"label": "Pets (cat/dog)", "value": "pets"},
                    {"label": "Vehicles (car/truck/bus)", "value": "vehicles"},
                    {"label": "Aircrafts (Airplane/Helicoptor)", "value": "airplane"}
                ],
                value="person",  # default
                style={"width": "40%"}
            )
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Input Data Type"),
            dcc.Dropdown(
                id="input-type",
                options=[
                    {"label": "Pictures", "value": "pictures"},
                    {"label": "Video", "value": "videos"}
                ],
                value="pictures",
                style={"width": "40%"}
            )
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Media Input Path (pictures or one video): "),
            dcc.Input(id="img-dir", type="text", value="./COCO_data/coco_images_people", style={"width": "40%"})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Output Data Path"),
            dcc.Input(id="out-dir", type="text", value="./output", style={"width": "40%"})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Model Confidence (0~1) :"),
            dcc.Input(id="conf-value", type="number", min=0, max=1, step=0.01, value=0.5, style={"width": "40%"})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Threshold (Object quantity in picture or frame exceeds)"),
            dcc.Input(id="threshold", type="number", min=0, value=1, style={"width": "40%"})
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Button("START DETECTION", id="run-button", n_clicks=0)
        ], style={"textAlign": "left", "marginBottom": "50px"}),
    ]),

    dcc.Graph(id="person-histogram"),

    html.Hr(),
    html.Div(id="result-area", children=[])
])


# ------------------------------
# Run Detection on Click
# ------------------------------
@app.callback(
    [Output("result-area", "children"),
     Output("person-histogram", "figure")],
    Input("run-button", "n_clicks"),
    State("img-dir", "value"),
    State("out-dir", "value"),
    State("conf-value", "value"),
    State("threshold", "value"),
    State("target-type", "value"),
    State("input-type", "value")
)
def run_detection(n_clicks, img_dir, out_dir, conf, threshold, target_type, input_type):
    if n_clicks == 0:
        return []
    
    temp_frame_dir = "tempFrame"
    if input_type == "videos":
        os.makedirs(temp_frame_dir, exist_ok=True)
        extract_frames_from_video(img_dir, temp_frame_dir)
        img_dir_to_use = temp_frame_dir
    else:
        img_dir_to_use = img_dir

    info_path = os.path.join(out_dir, "output_info.json")
    result_info, result_info_2 = detect_objects(img_dir_to_use, out_dir, info_path, conf, threshold, target_type)

    # === Generate Hist ===
    person_counts = [info["num_persons"] for info in result_info_2]
    count_hist = {}
    for count in person_counts:
        count_hist[count] = count_hist.get(count, 0) + 1

    # Turn to Graph in wedpage
    import plotly.graph_objs as go
    hist_fig = go.Figure(
        data=[go.Bar(x=list(count_hist.keys()), y=list(count_hist.values()))],
        layout_title_text="Object quantity distribution"
    )

    display_components = []
    for info in result_info:
        image_path = os.path.join(out_dir, info["file_name"])
        encoded_img = encode_image(image_path)
        display_components.append(html.Div([
            html.H4(f"{info['file_name']} - Detected Object Quantity: {info['num_persons']}"),
            html.Img(src=encoded_img, style={"width": "40%", "marginBottom": "30px"})
        ]))

    return display_components, hist_fig


if __name__ == '__main__':
    try:
        app.run(debug=True)  # For newer Dash 2.x+
    except AttributeError:
        app.run_server(debug=True)
