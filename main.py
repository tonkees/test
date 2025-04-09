import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import io

st.set_page_config(page_title="OCR на испанском", layout="centered")

st.title("Распознай испанский текст с изображения")

uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    if st.button("🔍 Распознать текст"):
        st.info("Распознавание текста...")

        reader = easyocr.Reader(['es'], gpu=False)
        result = reader.readtext(np.array(image), detail=0)

        full_text = " ".join(result)

        st.subheader("📝 Распознанный текст:")
        st.write(full_text)
        st.download_button(
            label="💾 Скачать как .txt",
            data=full_text,
            file_name="raspoznannyi_tekst.txt",
            mime="text/plain"
        )
else:
    st.markdown("⬆ Загрузите изображение, чтобы начать.")
