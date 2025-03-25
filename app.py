import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Ringan Bows", layout="wide")

# Judul aplikasi
st.title("📊 Analisis Data Ringan Bows Fakultas Peternakan")
st.markdown("Upload file Excel atau CSV kamu untuk melihat data dan analisis sederhana (bisa input data menggunakan 2 cara Excel atau Manual).")

# Upload file
uploaded_file = st.file_uploader("📁 Unggah file (.csv atau .xlsx)", type=['csv', 'xlsx'])

# Cek apakah ada file yang diunggah
if uploaded_file is not None:
    try:
        # Cek tipe file dan baca isinya
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Tampilkan preview data
        st.success("✅ File berhasil dimuat!")
        st.subheader("🔍 Preview Data")
        st.dataframe(df)

        # Statistik deskriptif
        st.subheader("📈 Statistik Deskriptif")
        st.write(df.describe())

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file untuk memulai analisis.")

# ---------------------------------------------
# Step 2 (Revisi): Input Data Manual - Dinamis

st.markdown("---")
st.header("📘 Input Data Manual (Dinamis)")

# Input jumlah kolom
jumlah_kolom = st.number_input("Berapa banyak kolom yang ingin kamu buat?", min_value=1, max_value=20, value=3, step=1)

# Inisialisasi nama kolom
if "nama_kolom_manual" not in st.session_state:
    st.session_state.nama_kolom_manual = [f"Kolom_{i+1}" for i in range(jumlah_kolom)]

# Form untuk masukkan nama kolom
with st.form("form_nama_kolom"):
    st.subheader("📝 Masukkan Nama Kolom")
    nama_kolom_baru = []
    for i in range(jumlah_kolom):
        kolom = st.text_input(f"Nama kolom ke-{i+1}", value=st.session_state.nama_kolom_manual[i] if i < len(st.session_state.nama_kolom_manual) else f"Kolom_{i+1}")
        nama_kolom_baru.append(kolom)
    simpan_nama = st.form_submit_button("✔️ Simpan Nama Kolom")

if simpan_nama:
    st.session_state.nama_kolom_manual = nama_kolom_baru

# Form untuk input data baris per baris
with st.form("form_input_data"):
    st.subheader("📊 Masukkan Data Tiap Baris")
    input_data = {}
    for nama in st.session_state.nama_kolom_manual:
        input_data[nama] = st.text_input(f"{nama}")
    tambah_baris = st.form_submit_button("+ Tambah ke Tabel")

# Inisialisasi penyimpanan tabel manual
if "data_manual" not in st.session_state:
    st.session_state.data_manual = pd.DataFrame(columns=st.session_state.nama_kolom_manual)

# Tambahkan baris jika diklik
if tambah_baris:
    new_row = {nama: input_data[nama] for nama in st.session_state.nama_kolom_manual}
    st.session_state.data_manual = pd.concat([
        st.session_state.data_manual,
        pd.DataFrame([new_row])
    ], ignore_index=True)
    st.success("✅ Baris berhasil ditambahkan!")

# Tampilkan tabel hasil input
if not st.session_state.data_manual.empty:
    st.subheader("📊 Tabel Data Manual")
    st.dataframe(st.session_state.data_manual)

# ---------------------------------------------
# Step 3: Pilih Analisis (T-Test, ANOVA, Korelasi, ROI, CTR)

from scipy.stats import ttest_ind, f_oneway, pearsonr

st.markdown("---")
st.header("⚙️ Pilih Jenis Analisis")

opsi_analisis = st.selectbox("Pilih jenis analisis:", [
    "Tidak ada", "Statistik: T-Test", "Statistik: ANOVA", "Statistik: Korelasi", "Marketing: ROI", "Marketing: CTR"
])

kolom_tersedia = st.session_state.data_manual.columns.tolist()

# ------------------ T-Test ------------------
if opsi_analisis == "Statistik: T-Test":
    st.subheader("📌 T-Test")
    if len(kolom_tersedia) >= 2:
        kolom_1 = st.selectbox("Pilih Kolom 1 (Grup A)", kolom_tersedia)
        kolom_2 = st.selectbox("Pilih Kolom 2 (Grup B)", kolom_tersedia)
        if st.button("🔍 Jalankan T-Test"):
            try:
                data1 = pd.to_numeric(st.session_state.data_manual[kolom_1], errors='coerce').dropna()
                data2 = pd.to_numeric(st.session_state.data_manual[kolom_2], errors='coerce').dropna()
                stat, p = ttest_ind(data1, data2)
                st.success("✅ T-Test Berhasil Dijalankan!")
                st.write(f"**Statistik T:** {stat}")
                st.write(f"**P-Value:** {p}")
                st.info("📌 Kesimpulan: " + ("Terdapat perbedaan signifikan." if p < 0.05 else "Tidak terdapat perbedaan signifikan."))
            except Exception as e:
                st.error(f"❌ Gagal menghitung T-Test: {e}")

# ------------------ ANOVA ------------------
elif opsi_analisis == "Statistik: ANOVA":
    st.subheader("📌 ANOVA (One Way)")
    grup = st.selectbox("Pilih kolom Grup Kategori", kolom_tersedia)
    nilai = st.selectbox("Pilih kolom Nilai", kolom_tersedia)
    if st.button("🔍 Jalankan ANOVA"):
        try:
            df = st.session_state.data_manual.dropna(subset=[grup, nilai])
            df[nilai] = pd.to_numeric(df[nilai], errors='coerce')
            grups = df.groupby(grup)[nilai].apply(list)
            stat, p = f_oneway(*grups)
            st.success("✅ ANOVA Berhasil Dijalankan!")
            st.write(f"**Statistik F:** {stat}")
            st.write(f"**P-Value:** {p}")
            st.info("📌 Kesimpulan: " + ("Ada perbedaan signifikan." if p < 0.05 else "Tidak ada perbedaan signifikan."))
        except Exception as e:
            st.error(f"❌ Gagal menghitung ANOVA: {e}")

# ------------------ Korelasi ------------------
elif opsi_analisis == "Statistik: Korelasi":
    st.subheader("📌 Korelasi Pearson")
    kolom_1 = st.selectbox("Pilih Kolom X", kolom_tersedia)
    kolom_2 = st.selectbox("Pilih Kolom Y", kolom_tersedia)
    if st.button("🔍 Hitung Korelasi"):
        try:
            x = pd.to_numeric(st.session_state.data_manual[kolom_1], errors='coerce').dropna()
            y = pd.to_numeric(st.session_state.data_manual[kolom_2], errors='coerce').dropna()
            r, p = pearsonr(x, y)
            st.success("✅ Korelasi Berhasil Dihitung!")
            st.write(f"**Koefisien Korelasi (r):** {r}")
            st.write(f"**P-Value:** {p}")
        except Exception as e:
            st.error(f"❌ Gagal menghitung Korelasi: {e}")

# ------------------ ROI ------------------
elif opsi_analisis == "Marketing: ROI":
    st.subheader("💸 Hitung ROI")
    with st.form("form_roi"):
        biaya = st.number_input("Total Biaya (Rp)", min_value=0.0)
        pendapatan = st.number_input("Total Pendapatan (Rp)", min_value=0.0)
        hitung_roi = st.form_submit_button("✔️ Hitung ROI")
    if hitung_roi:
        roi = ((pendapatan - biaya) / biaya) * 100 if biaya != 0 else 0
        st.success(f"ROI = {roi:.2f}%")
        st.info("📈 Hasil: " + ("Menguntungkan" if roi > 0 else "Impas" if roi == 0 else "Merugi"))

# ------------------ CTR ------------------
elif opsi_analisis == "Marketing: CTR":
    st.subheader("📊 Hitung CTR")
    with st.form("form_ctr"):
        impresi = st.number_input("Jumlah Tayangan (Impressions)", min_value=0)
        klik = st.number_input("Jumlah Klik", min_value=0)
        hitung_ctr = st.form_submit_button("✔️ Hitung CTR")
    if hitung_ctr:
        ctr = (klik / impresi) * 100 if impresi != 0 else 0
        st.success(f"CTR = {ctr:.2f}%")
        st.info("📌 Click Through Rate menunjukkan efektivitas iklan atau konten.")

# ---------------------------------------------
# Step 4: Visualisasi Grafik

import altair as alt

if not st.session_state.data_manual.empty:
    st.markdown("---")
    st.header("📈 Visualisasi Data")

    kolom_x = st.selectbox("Pilih kolom X (horizontal)", st.session_state.data_manual.columns)
    kolom_y = st.selectbox("Pilih kolom Y (vertical)", st.session_state.data_manual.columns)
    jenis_chart = st.selectbox("Pilih jenis grafik", ["Bar Chart", "Line Chart", "Scatter Plot"])

    try:
        df_viz = st.session_state.data_manual.copy()
        df_viz[kolom_x] = pd.to_numeric(df_viz[kolom_x], errors='coerce')
        df_viz[kolom_y] = pd.to_numeric(df_viz[kolom_y], errors='coerce')

        chart = None
        if jenis_chart == "Bar Chart":
            chart = alt.Chart(df_viz).mark_bar().encode(x=kolom_x, y=kolom_y)
        elif jenis_chart == "Line Chart":
            chart = alt.Chart(df_viz).mark_line().encode(x=kolom_x, y=kolom_y)
        elif jenis_chart == "Scatter Plot":
            chart = alt.Chart(df_viz).mark_circle(size=60).encode(x=kolom_x, y=kolom_y)

        if chart:
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membuat grafik: {e}")
