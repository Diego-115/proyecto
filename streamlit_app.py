import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from PIL import Image
import gdown
import os
import asyncio
import tempfile




# Function to create a card
def create_card(title, image_url):
    card_html = f"""
    <div class="card">
        <img class="card-image" src="{image_url}" alt="{title}">
        <div class="card-title">{title}</div>
    </div>
    """
    return card_html

# Function to download the model from Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)


# Cache the model loading
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    gdrive_url = 'https://drive.google.com/file/d/1dj66xwGVw6r4NM-d9u29NKEfSYREXQUk/view?usp=sharing'
    if not os.path.exists(model_path):
        download_model_from_gdrive(gdrive_url, model_path)
    model = YOLO(model_path)
    return model
model = load_model()


# CLASES DE LA APP

classes = [
    'Explosión','Mancha-marrón','Tizón-bacteriano','Tungro'
]

detected_classes = set()
def get_class_html(cls, detected_classes):
    detected_style = 'background-color:black; padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; color:black;'
    default_style = 'padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; background-color:black; color:white;'
    style = detected_style if cls in detected_classes else default_style
    return f'<span style="{style}">{cls}</span>'



# CLASE PARA CAPTURAR FRAME POR FRAME DEL VIDEO

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence = 0.25

    def set_params(self, model, confidence):
        self.model = model
        self.confidence = confidence

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.model:
            results = self.model(img_rgb, conf=self.confidence)
            if results:
                annotated_frame = results[0].plot()
                return av.VideoFrame.from_ndarray(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

async def process_image(image, model, confidence):
    img = Image.open(image)
    results = await asyncio.to_thread(model, img, conf=confidence)
    return results


#  APLICACION DE INICIO

def main():
    st.markdown("""
    <style>
    .card {
        background-color:#green;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        text-align: center;
        width: 90%;
        max-width: 300px;  /* Limit width for smaller screens */
        margin: 10px auto;  /* Center cards on the screen */
    }
    .card-title {
        font-size: 1.2em;  /* Slightly smaller font size */
        margin-bottom: 10px;
        color:green;
        background-color:#green;
    }
    .card-image {
        width: 100%;  /* Use full width of the card */
        height: auto;  /* Adjust height to maintain aspect ratio */
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

    st.title("Detección de Enfermedades")
    activities = ["Principal", "Usar cámara", "Subir imagen", "Subir vídeo"]
    choice = st.sidebar.selectbox("Selecciona actividad", activities)
    st.sidebar.markdown('---')

#   MODULO INICIAL DE LA APP

    if choice == "Principal":
        st.markdown("<h4 style='color:black;'>Aplicación web de detección de enfermedades en la hoja de arroz usando, Google Colab, Roboflow, Streamlit y lenguaje de programación Python.</h4><br>", unsafe_allow_html=True)
        html_classesp = [get_class_html(cls, detected_classes) for cls in classes]
        st.markdown(f"<div style='padding:4px; border: 2px solid black; border-radius: 10px;'><h4 style='color:black;text-align:center;'>4 Clases</h4><p style='color:black;text-align:center;'>{' '.join(html_classesp)}</p></div><br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.markdown(create_card("Usar cámara", "https://cdn-icons-png.freepik.com/256/3004/3004613.png?semt=ais_hybrid"), unsafe_allow_html=True)
        col2.markdown(create_card("Subir imagen", "https://cdn-icons-png.freepik.com/256/16611/16611551.png?semt=ais_hybrid"), unsafe_allow_html=True)
        col3.markdown(create_card("Subir vídeo", "https://cdn-icons-png.freepik.com/256/1179/1179120.png?semt=ais_hybrid"), unsafe_allow_html=True)

# SI SE CELECCIONA LA CAMARA

    elif choice == "Usar cámara":
            st.header("Utiliza tu cámara")
            if model:
                confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
                start_detection = st.checkbox("Iniciar detección de objetos")
                video_transformer = VideoTransformer()
                if start_detection:
                    video_transformer.set_params(model, confidence_slider)
                webrtc_streamer(
                    key="example",
                    video_processor_factory=lambda: video_transformer,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
                        "bundlePolicy": "max-bundle"
                    },
                    media_stream_constraints={"video": True, "audio": False},
                )        



# SI SE SELECCIONA SUBIR UNA IMAGEN

    elif choice == "Subir imagen":
        confidence_slider = st.sidebar.slider('Confianza', min_value=0.0, max_value=1.0, value=0.25, key='imagen')   

        html_classes = [get_class_html(cls, detected_classes) for cls in classes]
        text_placeholder = st.empty()
        text_placeholder.markdown(f"<div style='padding:4px; border: 2px solid black; border-radius: 10px;'><h4 style='color:black;text-align:center;'>4 Clases</h4><p style='color:black;text-align:center;'>{' '.join(html_classes)}</p></div>", unsafe_allow_html=True)
        change_text = st.checkbox("Objetos Detectados")
        image = st.file_uploader('Sube imagen', type=['png', 'jpg', 'jpeg', 'gif'])

        if image:
            col1, col2, col3 = st.columns([1, 1, 1])
            col1.image(image, caption='Imagen original', use_column_width=True)  # Ajusta el uso del ancho de la columna
            if model:
                with col2:
                    with st.spinner('Procesando imagen...'):
                        results = asyncio.run(process_image(image, model, confidence_slider))
                        if results:
                            annotated_frame = results[0].plot()
                            annotated_frame = cv2.cvtColor(annotated_frame,cv2.COLOR_RGB2BGR)
                            col2.image(annotated_frame, caption='Imagen anotada', use_column_width=True)  # Ajusta el uso del ancho de la columna
                            for result in results[0].boxes:
                                idx = int(result.cls.cpu().numpy()[0])
                                confidence = result.conf.cpu().numpy()[0]
                                detected_class = classes[idx]
                                detected_classes.add(detected_class)
                                col3.markdown(f"<div style='background-color:white;padding:5px;border-radius:5px;margin:5px 0; color:black;'><b>Clase:</b> <span style='color:black'>{detected_class}</span><br><b>Confianza:</b> {confidence:.2f}<br></div>", unsafe_allow_html=True)
                                st.balloons()
                        else:
                            col3.write("No se detectaron objetos.")

            else:
                st.error("Model is not loaded. Please check the logs for errors.")
        
# SI SE CELECCIONA SUBRI UN VIDEO

    elif choice == "Subir vídeo":
        st.header("Sube un vídeo para detección de objetos")
        confidence_slider = st.sidebar.slider('Confianza', min_value=0.0, max_value=1.0, value=0.25, key='video')
        video_file = st.file_uploader('Sube vídeo', type=['mp4', 'avi', 'mov', 'mkv'])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            col1, col2 = st.columns(2)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
            stframe = st.empty()
            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with col1:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if model:
                        results = model(img_rgb, conf=confidence_slider)
                        if results:
                            annotated_frame = results[0].plot()
                            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                            stframe.image(annotated_frame, channels="RGB", use_column_width=True)  # Ajusta el uso del ancho de la columna
                    progress_bar.progress(min(cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count, 1.0))
                cap.release()
                out.release()
                st.success('Procesamiento de vídeo completo.')
                st.balloons()
 #           with col2:
        #       st.video('output.mp4')
#           with open('output.mp4', 'rb') as f:
  #              video_data = f.read()
   #         st.markdown("Descargar video procesado:")
    #    st.download_button('Descargar', video_data, 'output.mp4', 'video/mp4')


if __name__ == "__main__":
    main()


