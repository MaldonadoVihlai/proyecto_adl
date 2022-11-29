import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
import streamlit as st
import constants, keras, re, glob, pydicom, cv2, os


@st.experimental_singleton()
def load_model():
    return keras.models.load_model('model/' + constants.MODEL)


model = load_model()


def get_top_page_content(st):
    st.image(constants.IMAGE_BANNER)
    st.title(
        'Detección del estado de metilación de la enzima MGMT en tumores cerebrales')
    st.markdown('**Nota**: Cargue la carpeta correspondiente a un **único** paciente.\
         Recuerde que la carpeta debe contener únicamente archivos \
        dcm.')


def load_dicom_image(path, img_size=constants.SIZE):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_images_3d(scan_id, num_imgs=constants.NUM_IMAGES, img_size=constants.SIZE, mri_type="T2w"):
    files = sorted(glob.glob(constants.EXTRACTION_DIRECTORY+f"/{scan_id}/{mri_type}/*.dcm"),
                   key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    middle = len(files)//2
    num_imgs2 = num_imgs//2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    img3d = np.stack([load_dicom_image(f) for f in files[p1:p2]]).T
    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
        img3d = np.concatenate((img3d,  n_zero), axis=-1)
    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)
    return img3d


def generator_train():
    x = load_dicom_images_3d(
        next(os.walk(constants.EXTRACTION_DIRECTORY))[1][0])
    yield x


def build_dataset():
    ds_train = tf.data.Dataset.from_generator(generator_train, args=[],
                                              output_types=tf.int16,
                                              output_shapes=(constants.SIZE, constants.SIZE, constants.NUM_IMAGES))
    ds_train = ds_train.batch(constants.BATCH_SIZE)
    return ds_train


def pred_dataset(ds_train, model):
    y_pred = list()

    for images in ds_train:
        y_pred.extend(model.predict(
            images, verbose=0).astype(float).tolist())
    return y_pred


def process_dataset():
    ds_train = build_dataset()
    return pred_dataset(ds_train, model)

def get_mgmt_state(state):
    return '### **De acuerdo al modelo el paciente cuenta con una probabilidad  de '+ str(round(state, 3)) + ' de tener la enzima MGMT**'


def st_directory_picker(initial_path=Path()):

    st.markdown("#### Selección de la carpeta")

    if "path" not in st.session_state:
        st.session_state.path = initial_path.absolute()

    manual_input = st.text_input(
        "Cargue la carpeta con las imágenes:", st.session_state.path)

    manual_input = Path(manual_input)
    if manual_input != st.session_state.path:
        st.session_state.path = manual_input
        st.experimental_rerun()

    _, col1, col2, col3, _ = st.columns([3, 1, 3, 1, 3])

    with col1:
        st.markdown("#")
        if st.button("⬅️") and "path" in st.session_state:
            st.session_state.path = st.session_state.path.parent
            st.experimental_rerun()

    with col2:
        subdirectroies = [
            f.stem
            for f in st.session_state.path.iterdir()
            if f.is_dir()
            and (not f.stem.startswith(".") and not f.stem.startswith("__"))
        ]
        if subdirectroies:
            st.session_state.new_dir = st.selectbox(
                "Subdirectorios", sorted(subdirectroies)
            )
        else:
            st.markdown("#")
            st.markdown(
                "<font color='#FF0000'>No subdir</font>", unsafe_allow_html=True
            )

    with col3:
        if subdirectroies:
            st.markdown("#")
            if st.button("➡️") and "path" in st.session_state:
                st.session_state.path = Path(
                    st.session_state.path, st.session_state.new_dir
                )
                st.experimental_rerun()

    return st.session_state.path
