import gradio as gr
import cv2

from cnn2snn import set_akida_version, AkidaVersion
import akida

import numpy as np

import time

import plotly.graph_objects as go

def create_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={'axis': {'range': [0, 30]}},
        domain={'x': [0, 1], 'y': [0, 1]},
    ))
    fig.update_layout(width=400, height=300)
    return fig

# Softmax for an array of values
def softmaxArray(values):
    # Assuming array shape is (1, 1, 1, x), flatten to get the values
    values = values.ravel()
    exp_values = np.exp(values)
    sum_exp = np.sum(exp_values)
    softmax_values = exp_values / sum_exp
    return softmax_values


# Decode numeric labels into human readable ones: contains all string names for classes
# available in the dataset
import os
import csv
# Get current working directory
current_directory = os.getcwd()
# Construct full file path
file_path = os.path.join(current_directory, 'datasets/jester_subset', 'jester-v1-labels.csv')
# Open the file
with open(file_path, 'r') as csvfile:
    labels = [row[0] for row in csv.reader(csvfile)]


image_x = 100
image_y = 100
image_z = 3
def decodeOutput(inp):
        global akida_model
        inp = cv2.resize(inp, (image_x, image_y))
        inp = inp.reshape((-1, image_x, image_y, image_z))
        timer_start = time.time()
        predictions = softmaxArray(akida_model.predict(inp))
        frame_time = time.time() - timer_start
        fps = 1 / frame_time if frame_time > 0 else 0
        confidences = {labels[i]: predictions[i] for i in range(len(predictions))}
        sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[0]))

        # Get top prediction and send to AFrame
        top_gesture = max(confidences, key=confidences.get)
        confidence = confidences[top_gesture]

        gesture_key = top_gesture
        #print(f"gesture_key: {gesture_key}, confidence: {confidence}, fps: {fps}")
        gesture_prediction = ""

        #if confidence > 0.84:
        if confidence > 0.7:
                gesture_to_key = {
                'Swiping Left': 'ArrowLeft',
                'Swiping Right': 'ArrowRight',
                'Zooming In With Two Fingers': 'ArrowUp', 
                'Zooming Out With Two Fingers': 'ArrowDown',  # Commented out to avoid conflict with zooming in
                'Shaking Hand': 'Space'
                }
                gesture_key = gesture_to_key.get(top_gesture, "")
                gesture_prediction = top_gesture
                #print(f"Gesture key: {gesture_key}")
        else:
                gesture_key = ""
                gesture_prediction = ""
                # print("No valid gesture detected")
                
                

        return sorted_confidences, fps, gesture_key, gesture_prediction

def classify_image(inp):

  confidences, fps, gesture_key, gesture_prediction = decodeOutput(inp)
  #print(f"Gesture: {gesture_key}, FPS: {fps}")

  return create_gauge(round(fps, 2)), gesture_key, gesture_prediction

from akida_models.model_io import load_model
akida_model = load_model("models/tenn_spatiotemporal_jester_buffer_i8_w8_a8.fbz")

with set_akida_version(AkidaVersion.v2):
            devices = akida.devices()
            if len(devices) > 0:
                print(f'Available devices: {[dev.desc for dev in devices]}')
                device = devices[0]
                print(device.version)
                try:
                    akida_model.map(device)
                    print(f"Mapping to Akida device {device.desc}.")
                    mappedDevice = device.version
                except Exception as e:
                    print("Model not compatible with FPGA. Running on CPU.")
                    mappedDevice = "CPU"
            else:
                print("No Akida devices found, running on CPU.")
                mappedDevice = "CPU"

theme = gr.themes.Base(
    text_size="sm",
    spacing_size="sm",
    radius_size="sm",
)

gr.set_static_paths(paths=["vr/", "img/"])


with gr.Blocks(
    title="Brainchip",
    fill_width=True,
    fill_height=True,
    delete_cache=[180, 600],
    theme=theme
) as gesture_demo:
    gr.Markdown("""
        <h1 style="text-align: center;">Akida Cloud</h1>
        <br>
        """)
    with gr.Row():
        gr.Markdown("## Gesture Recognition with RGB Camera")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], type="numpy")
            gr.Markdown("**ℹ️ Please press the 'Record' button to start inference.**")
            gesture_prediction = gr.Textbox(visible=True, label="Gesture", placeholder="Detected gesture will appear here")
        with gr.Column():
            gr.Markdown(f"""Device: {mappedDevice}""")
            js_output = gr.HTML(visible=False)
            gesture_output = gr.Textbox(visible=False)
            aframe_component = gr.HTML("""
                <iframe src="/gradio_api/file=vr/photo_gallery.html" 
                        width="800" height="600" 
                        frameborder="0"
                        id="aframe-iframe">
                </iframe>
            """)
            #output_label = gr.Label(elem_classes=labels, visible=False, render=False, label="Classification Results")
            #output_dataframe = gr.DataFrame(label="Classification Results", headers=["Label", "Confidence"]) 
            #print(output_label)
            plot = gr.Plot(label="Frames per second")
        dep = input_img.stream(classify_image, [input_img], [plot, gesture_output, gesture_prediction],
                                time_limit=30, stream_every=0.1, concurrency_limit=30)      
        gesture_output.change(
            fn=None,
            js="""(gesture) => {
                                    console.log('Gesture key:', gesture);
                                    // Send gesture to AFrame iframe
                                    if (gesture && gesture.trim() !== '') {
                                        console.log('Sending gesture during stream:', gesture);
                                        let iframe = document.getElementById('aframe-iframe');
                                        if (iframe) {
                                            iframe.contentWindow.postMessage({
                                                type: 'gesture',
                                                keyCode: gesture
                                            }, '*');
                                        }
                                    }
                                    return outputs;
                                }""",
            inputs=[gesture_output], outputs=[]
        )
        


if __name__ == "__main__":
    gesture_demo.launch()