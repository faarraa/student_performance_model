from pathlib import Path
import numpy as np
import streamlit as st
import torch
from model import StudentPerformanceANN

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1150px}
.main-title{font-size:2.25rem;font-weight:800;letter-spacing:-1px;margin-bottom:.2rem}
.sub-title{color:#64748b;margin-bottom:1.6rem}
.panel{border:1px solid #e5e7eb;border-radius:16px;padding:22px;background:#fff;min-height:390px}
.result{border-radius:14px;padding:22px;background:#f8fafc;border:1px solid #e2e8f0}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">🎓 Student Performance Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Prediksi risiko performa akademik berdasarkan karakteristik siswa.</div>', unsafe_allow_html=True)

MODEL_PATH = Path("model/student_performance_model.pth")
if not MODEL_PATH.exists():
    st.error("File model .pth belum ditemukan. Jalankan `python train_model.py` terlebih dahulu.")
    st.stop()

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    net = StudentPerformanceANN(checkpoint["input_size"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, checkpoint

model, checkpoint = load_model()
categories = checkpoint["categories"]

with st.container():
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("### Data Siswa")
        st.caption("Pilih karakteristik siswa untuk melakukan prediksi.")
        gender = st.selectbox("Gender", categories["gender"], format_func=lambda x: "Perempuan" if x == "female" else "Laki-laki")
        race = st.selectbox("Race / Ethnicity", categories["race/ethnicity"])
        parent = st.selectbox("Pendidikan Orang Tua", categories["parental level of education"])
        lunch = st.selectbox("Jenis Makan Siang", categories["lunch"], format_func=lambda x: "Standar" if x == "standard" else "Gratis / Subsidi")
        prep = st.selectbox("Kursus Persiapan Ujian", categories["test preparation course"], format_func=lambda x: "Sudah selesai" if x == "completed" else "Belum")
        predict = st.button("Prediksi Performa", type="primary", use_container_width=True)
    with right:
        st.markdown("### Hasil Prediksi")
        st.caption("Hasil akan muncul setelah data diproses oleh model ANN.")
        if predict:
            values = [gender, race, parent, lunch, prep]
            vector = []
            for feature, value in zip(checkpoint["features"], values):
                vector.extend([1.0 if value == category else 0.0 for category in categories[feature]])
            with torch.no_grad():
                probability = torch.sigmoid(model(torch.tensor([vector], dtype=torch.float32))).item()
            label = int(probability >= checkpoint.get("threshold", 0.5))
            st.markdown('<div class="result">', unsafe_allow_html=True)
            if label == 1:
                st.error("Performa Rendah")
                st.write("Model memperkirakan siswa berisiko memiliki rata-rata nilai ujian di bawah 60.")
            else:
                st.success("Tidak Rendah")
                st.write("Model memperkirakan siswa tidak termasuk kategori performa rendah.")
            st.metric("Probabilitas Performa Rendah", f"{probability * 100:.1f}%")
            st.progress(float(probability))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Isi data di sebelah kiri, lalu klik **Prediksi Performa** untuk melihat hasil.")
            st.metric("Ambang Kategori", "60")
            st.write("Kategori rendah ditentukan dari rata-rata nilai matematika, membaca, dan menulis yang kurang dari 60.")

st.divider()
st.caption("Catatan: Prediksi ini merupakan estimasi berdasarkan pola dataset, bukan penilaian atau keputusan akademik yang pasti.")
