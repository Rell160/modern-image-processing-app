import streamlit as st
import cv2
import numpy as np
import random

from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Digital Image Processing App",
    page_icon="📸",
    layout="wide"
)

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0E1117;
    }

    h1, h2, h3, h4 {
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("📸 Aplikasi Pengolah Citra Digital Modern")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def ensure_bgr(image):

    if len(image.shape) == 2:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    return image


def add_film_grain(image, intensity=15):

    noise = np.random.normal(
        0,
        intensity,
        image.shape
    ).astype(np.int16)

    grain = image.astype(np.int16) + noise

    grain = np.clip(
        grain,
        0,
        255
    )

    return grain.astype(np.uint8)


def add_dust(image, amount=100):

    dust = image.copy()

    h, w = dust.shape[:2]

    for _ in range(amount):

        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        radius = random.randint(1, 3)

        cv2.circle(
            dust,
            (x, y),
            radius,
            (255, 255, 255),
            -1
        )

    return dust


def add_scratches(image, amount=100):

    scratch = image.copy()

    h, w = scratch.shape[:2]

    for _ in range(amount):

        x = random.randint(0, w - 1)

        y1 = random.randint(0, h)
        y2 = random.randint(0, h)

        cv2.line(
            scratch,
            (x, y1),
            (x, y2),
            (255, 255, 255),
            1
        )

    return scratch


def add_chromatic_aberration(image):

    b, g, r = cv2.split(image)

    rows, cols = image.shape[:2]

    M1 = np.float32([
        [1, 0, -2],
        [0, 1, 0]
    ])

    M2 = np.float32([
        [1, 0, 2],
        [0, 1, 0]
    ])

    r_shifted = cv2.warpAffine(
        r,
        M1,
        (cols, rows)
    )

    b_shifted = cv2.warpAffine(
        b,
        M2,
        (cols, rows)
    )

    return cv2.merge([
        b_shifted,
        g,
        r_shifted
    ])


@st.cache_data
def create_light_leak(image, mode="warm"):

    h, w = image.shape[:2]

    base = image.astype(np.float32)

    overlay = np.zeros(
        (h, w, 3),
        dtype=np.float32
    )

    positions = [

        (0, h // 2),
        (w, h // 2),
        (w // 2, 0),
        (w // 2, h)

    ]

    center_x, center_y = random.choice(positions)

    if mode == "warm":

        colors = [

            (0, 120, 255),
            (0, 180, 255),
            (80, 200, 255)

        ]

    else:

        colors = [

            (255, 180, 0),
            (255, 255, 0),
            (255, 120, 50)

        ]

    for radius in range(700, 0, -30):

        alpha = radius / 700

        color = random.choice(colors)

        glow_color = (
            color[0] * alpha,
            color[1] * alpha,
            color[2] * alpha
        )

        cv2.circle(
            overlay,
            (center_x, center_y),
            radius,
            glow_color,
            -1
        )

    overlay = cv2.GaussianBlur(
        overlay,
        (251, 251),
        0
    )

    overlay *= 0.45

    screen = 255 - (
        (255 - base) *
        (255 - overlay) / 255
    )

    screen = np.clip(
        screen,
        0,
        255
    )

    return screen.astype(np.uint8)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎨 Filter Menu")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "webp"]
)

# =========================================================
# RESET BUTTON
# =========================================================

if st.sidebar.button("🔄 Reset Filter"):

    st.session_state.selected_filters = []
    st.session_state.brightness = 0
    st.session_state.contrast = 1.0
    st.session_state.filter_intensity = 1.0
    st.rerun()

# =========================================================
# FILTER SELECTION
# =========================================================

selected_filters = st.sidebar.multiselect(

    "Pilih Filter",

    [

        "Grayscale",
        "Threshold",
        "Gaussian Blur",
        "Canny Edge",
        "Rotate 90°",
        "Flip Horizontal",
        "Flip Vertical",
        "Vintage",
        "Faded Film",
        "Old Photo",
        "Light Leak Warm",
        "Light Leak Cool",
        "Scratches"

    ],

    key="selected_filters"
)

brightness = st.sidebar.slider(
    "Brightness",
    -100,
    100,
    0,
    key="brightness"
)

contrast = st.sidebar.slider(
    "Contrast",
    0.5,
    3.0,
    1.0,
    key="contrast"
)

filter_intensity = st.sidebar.slider(
    "Filter Intensity",
    0.1,
    2.0,
    1.0,
    key="filter_intensity"
)

# =========================================================
# BEFORE AFTER SLIDER
# =========================================================

before_after_slider = st.sidebar.slider(
    "Before / After Slider",
    0,
    100,
    50
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    🎓 Digital Image Processing App

    Features:
    - Multiple Filter Stacking
    - Realtime Preview
    - Before / After Slider
    - Reset Filter
    - Vintage Effects
    - Light Leak Effects
    - Brightness & Contrast

    Built with:
    - Streamlit
    - OpenCV
    - NumPy
    """
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

    except Exception:

        st.error("❌ Gagal membuka gambar")
        st.stop()

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    processed = image.copy()

    # =====================================================
    # MULTIPLE FILTER STACKING
    # =====================================================

    for filter_name in selected_filters:

        # =================================================
        # GRAYSCALE
        # =================================================

        if filter_name == "Grayscale":

            processed = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2GRAY
            )

            st.success(
                "✅ Grayscale filter applied"
            )

        # =================================================
        # THRESHOLD
        # =================================================

        elif filter_name == "Threshold":

            gray = cv2.cvtColor(
                ensure_bgr(processed),
                cv2.COLOR_BGR2GRAY
            )

            _, processed = cv2.threshold(
                gray,
                127,
                255,
                cv2.THRESH_BINARY
            )

        # =================================================
        # GAUSSIAN BLUR
        # =================================================

        elif filter_name == "Gaussian Blur":

            processed = ensure_bgr(processed)

            ksize = int(9 * filter_intensity)

            if ksize % 2 == 0:
                ksize += 1

            if ksize < 3:
                ksize = 3

            processed = cv2.GaussianBlur(
                processed,
                (ksize, ksize),
                0
            )

        # =================================================
        # CANNY EDGE
        # =================================================

        elif filter_name == "Canny Edge":

            gray = cv2.cvtColor(
                ensure_bgr(processed),
                cv2.COLOR_BGR2GRAY
            )

            processed = cv2.Canny(
                gray,
                100,
                200
            )

        # =================================================
        # ROTATE
        # =================================================

        elif filter_name == "Rotate 90°":

            processed = cv2.rotate(
                processed,
                cv2.ROTATE_90_CLOCKWISE
            )

        # =================================================
        # FLIP HORIZONTAL
        # =================================================

        elif filter_name == "Flip Horizontal":

            processed = cv2.flip(
                processed,
                1
            )

        # =================================================
        # FLIP VERTICAL
        # =================================================

        elif filter_name == "Flip Vertical":

            processed = cv2.flip(
                processed,
                0
            )

        # =================================================
        # VINTAGE
        # =================================================

        elif filter_name == "Vintage":

            processed = ensure_bgr(processed)

            sepia_filter = np.array([

                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]

            ])

            sepia = cv2.transform(
                processed,
                sepia_filter
            )

            sepia = np.clip(
                sepia,
                0,
                255
            ).astype(np.uint8)

            processed = cv2.addWeighted(
                processed,
                1 - (filter_intensity / 2),
                sepia,
                filter_intensity / 2,
                0
            )

        # =================================================
        # FADED FILM
        # =================================================

        elif filter_name == "Faded Film":

            processed = cv2.convertScaleAbs(
                processed,
                alpha=0.8,
                beta=int(30 * filter_intensity)
            )

        # =================================================
        # OLD PHOTO
        # =================================================

        elif filter_name == "Old Photo":

            processed = ensure_bgr(processed)

            sepia_filter = np.array([

                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]

            ])

            processed = cv2.transform(
                processed,
                sepia_filter
            )

            processed = np.clip(
                processed,
                0,
                255
            ).astype(np.uint8)

            processed = add_film_grain(
                processed,
                int(15 * filter_intensity)
            )

            processed = add_dust(
                processed,
                int(200 * filter_intensity)
            )

        # =================================================
        # LIGHT LEAK WARM
        # =================================================

        elif filter_name == "Light Leak Warm":

            processed = ensure_bgr(processed)

            processed = create_light_leak(
                processed,
                "warm"
            )

        # =================================================
        # LIGHT LEAK COOL
        # =================================================

        elif filter_name == "Light Leak Cool":

            processed = ensure_bgr(processed)

            processed = create_light_leak(
                processed,
                "cool"
            )

            processed = add_chromatic_aberration(
                processed
            )

        # =================================================
        # SCRATCHES
        # =================================================

        elif filter_name == "Scratches":

            processed = ensure_bgr(processed)

            processed = add_scratches(
                processed,
                int(100 * filter_intensity)
            )

    # =====================================================
    # BRIGHTNESS & CONTRAST
    # =====================================================

    processed = cv2.convertScaleAbs(
        processed,
        alpha=contrast,
        beta=brightness
    )

    # =====================================================
    # BEFORE AFTER COMPARISON
    # =====================================================

    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    if len(processed.shape) == 2:

        processed_rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_GRAY2RGB
        )

    else:

        processed_rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
        )

    h, w = processed_rgb.shape[:2]

    split_position = int(
        (before_after_slider / 100) * w
    )

    comparison = processed_rgb.copy()

    comparison[:, :split_position] = original_rgb[:, :split_position]

    # =====================================================
    # DISPLAY
    # =====================================================

    st.subheader("🖼 Before / After Comparison")

    st.image(
        comparison,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Original Image")

        st.image(
            original_rgb,
            use_container_width=True
        )

    with col2:

        st.subheader("✨ Processed Image")

        st.image(
            processed_rgb,
            use_container_width=True
        )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    _, buffer = cv2.imencode(
        ".png",
        processed
    )

    st.download_button(
        "💾 Download Image",
        buffer.tobytes(),
        file_name="processed_image.png",
        mime="image/png"
    )