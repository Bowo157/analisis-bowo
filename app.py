import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, pearsonr
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Ringan Bows", layout="wide")

# Judul aplikasi
st.title("📊 Analisis Data Ringan Bows Fakultas Peternakan")
st.markdown("Upload file Excel atau CSV kamu untuk melihat data dan analisis sederhana (bisa input data menggunakan 2 cara Excel atau Manual).")

# Helper function untuk validasi data numerik
def validate_numeric_data(df, column):
    try:
        numeric_data = pd.to_numeric(df[column], errors='coerce')
        if numeric_data.isna().all():
            return False, f"Kolom {column} tidak memiliki data numerik yang valid"
        return True, numeric_data
    except Exception as e:
        return False, f"Error pada kolom {column}: {str(e)}"

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

        # Tampilkan preview data dengan informasi tipe data
        st.success("✅ File berhasil dimuat!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Preview Data")
            st.dataframe(df)
        with col2:
            st.subheader("📊 Informasi Data")
            st.write("Tipe data setiap kolom:")
            st.write(df.dtypes)
            st.write("Jumlah data:", len(df))
            st.write("Jumlah kolom:", len(df.columns))

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file untuk memulai analisis.")

# Input Data Manual
st.markdown("---")
st.header("📘 Input Data Manual")

# Input jumlah kolom dengan validasi
jumlah_kolom = st.number_input("Berapa banyak kolom yang ingin kamu buat?", min_value=1, max_value=20, value=3, step=1)

# Inisialisasi state
if "nama_kolom_manual" not in st.session_state:
    st.session_state.nama_kolom_manual = [f"Kolom_{i+1}" for i in range(jumlah_kolom)]
if "tipe_data_kolom" not in st.session_state:
    st.session_state.tipe_data_kolom = ["text" for _ in range(jumlah_kolom)]

# Form untuk nama dan tipe kolom
with st.form("form_nama_kolom"):
    st.subheader("📝 Masukkan Nama dan Tipe Kolom")
    nama_kolom_baru = []
    tipe_data_baru = []
    
    for i in range(jumlah_kolom):
        col1, col2 = st.columns([2,1])
        with col1:
            nama = st.text_input(
                f"Nama kolom ke-{i+1}",
                value=st.session_state.nama_kolom_manual[i] if i < len(st.session_state.nama_kolom_manual) else f"Kolom_{i+1}"
            )
            nama_kolom_baru.append(nama)
        with col2:
            tipe = st.selectbox(
                f"Tipe data ke-{i+1}",
                ["text", "number", "category"],
                index=0 if i >= len(st.session_state.tipe_data_kolom) else ["text", "number", "category"].index(st.session_state.tipe_data_kolom[i])
            )
            tipe_data_baru.append(tipe)
    
    simpan_nama = st.form_submit_button("✔️ Simpan Pengaturan Kolom")

if simpan_nama:
    st.session_state.nama_kolom_manual = nama_kolom_baru
    st.session_state.tipe_data_kolom = tipe_data_baru
    st.session_state.data_manual = pd.DataFrame(columns=nama_kolom_baru)
    st.success("✅ Pengaturan kolom berhasil disimpan!")

# Form untuk input data
with st.form("form_input_data"):
    st.subheader("📊 Masukkan Data Tiap Baris")
    input_data = {}
    
    for nama, tipe in zip(st.session_state.nama_kolom_manual, st.session_state.tipe_data_kolom):
        if tipe == "number":
            input_data[nama] = st.number_input(f"{nama}", value=0.0, format="%.2f")
        elif tipe == "category":
            input_data[nama] = st.selectbox(f"{nama}", ["Kategori A", "Kategori B", "Kategori C"])
        else:
            input_data[nama] = st.text_input(f"{nama}")
    
    tambah_baris = st.form_submit_button("+ Tambah ke Tabel")

# Inisialisasi penyimpanan tabel manual
if "data_manual" not in st.session_state:
    st.session_state.data_manual = pd.DataFrame(columns=st.session_state.nama_kolom_manual)

# Tambahkan baris jika diklik
if tambah_baris:
    st.session_state.data_manual = pd.concat([
        st.session_state.data_manual,
        pd.DataFrame([input_data])
    ], ignore_index=True)
    st.success("✅ Baris berhasil ditambahkan!")

# Tampilkan tabel hasil input
if not st.session_state.data_manual.empty:
    st.subheader("📊 Tabel Data Manual")
    st.dataframe(st.session_state.data_manual)

# Analisis Data
st.markdown("---")
st.header("⚙️ Analisis Data")

# Pilih dataset untuk analisis
dataset_source = st.radio("Pilih sumber data untuk analisis:",
                         ["Data dari File", "Data Manual"],
                         disabled=(uploaded_file is None and st.session_state.data_manual.empty))

# Pilih dataset yang akan dianalisis
if dataset_source == "Data dari File" and uploaded_file is not None:
    df_analysis = df
elif dataset_source == "Data Manual" and not st.session_state.data_manual.empty:
    df_analysis = st.session_state.data_manual
else:
    st.warning("⚠️ Silakan input data terlebih dahulu sebelum melakukan analisis.")
    st.stop()

# Pilih jenis analisis
analysis_type = st.selectbox("Pilih jenis analisis:", [
    "Tidak ada", "Statistik: T-Test", "Statistik: ANOVA", "Statistik: Korelasi",
    "Marketing: ROI", "Marketing: CTR"
])

if analysis_type != "Tidak ada":
    st.subheader(f"📊 {analysis_type}")
    
    if analysis_type == "Statistik: T-Test":
        col1, col2 = st.columns(2)
        with col1:
            kolom_1 = st.selectbox("Pilih Kolom 1 (Grup A)", df_analysis.columns)
        with col2:
            kolom_2 = st.selectbox("Pilih Kolom 2 (Grup B)", df_analysis.columns)
        
        if st.button("🔍 Jalankan T-Test"):
            # Validasi dan analisis
            valid1, data1 = validate_numeric_data(df_analysis, kolom_1)
            valid2, data2 = validate_numeric_data(df_analysis, kolom_2)
            
            if valid1 and valid2:
                # Analisis
                stat, p = ttest_ind(data1.dropna(), data2.dropna())
                effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                
                # Tampilkan hasil dalam tabel
                results_df = pd.DataFrame({
                    'Metrik': ['T-Statistic', 'P-Value', "Effect Size (Cohen's d)"],
                    'Nilai': [stat, p, effect_size]
                })
                st.table(results_df)
                
                # Interpretasi
                st.write("Interpretasi:")
                if p < 0.05:
                    st.success(f"✅ Terdapat perbedaan signifikan antara {kolom_1} dan {kolom_2}")
                else:
                    st.info(f"ℹ️ Tidak terdapat perbedaan signifikan antara {kolom_1} dan {kolom_2}")
                
                # Visualisasi
                fig = go.Figure()
                fig.add_trace(go.Box(y=data1.dropna(), name=kolom_1))
                fig.add_trace(go.Box(y=data2.dropna(), name=kolom_2))
                fig.update_layout(title=f"Perbandingan {kolom_1} vs {kolom_2}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Data harus berupa angka untuk analisis T-Test")
    
    elif analysis_type == "Statistik: ANOVA":
        col1, col2 = st.columns(2)
        with col1:
            grup = st.selectbox("Pilih kolom Grup/Kategori", df_analysis.columns)
        with col2:
            nilai = st.selectbox("Pilih kolom Nilai", df_analysis.columns)
        
        if st.button("🔍 Jalankan ANOVA"):
            valid, data = validate_numeric_data(df_analysis, nilai)
            
            if valid:
                # Analisis
                groups = df_analysis.groupby(grup)[nilai].apply(list)
                f_stat, p_val = f_oneway(*groups)
                
                # Effect size (Eta-squared)
                df_total = len(df_analysis) - 1
                df_between = len(groups) - 1
                ss_between = sum(len(group) * ((np.mean(group) - np.mean(data))**2) for group in groups)
                ss_total = sum((x - np.mean(data))**2 for x in data)
                eta_squared = ss_between / ss_total
                
                # Tampilkan hasil
                results_df = pd.DataFrame({
                    'Metrik': ['F-Statistic', 'P-Value', 'Eta-squared'],
                    'Nilai': [f_stat, p_val, eta_squared]
                })
                st.table(results_df)
                
                # Interpretasi
                st.write("Interpretasi:")
                if p_val < 0.05:
                    st.success("✅ Terdapat perbedaan signifikan antar grup")
                else:
                    st.info("ℹ️ Tidak terdapat perbedaan signifikan antar grup")
                
                # Visualisasi
                fig = go.Figure()
                for group_name, group_data in groups.items():
                    fig.add_trace(go.Violin(y=group_data, name=str(group_name), box_visible=True))
                fig.update_layout(title=f"Distribusi {nilai} berdasarkan {grup}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Kolom nilai harus berupa angka untuk analisis ANOVA")
    
    elif analysis_type == "Statistik: Korelasi":
        col1, col2 = st.columns(2)
        with col1:
            var_x = st.selectbox("Pilih Variabel X", df_analysis.columns)
        with col2:
            var_y = st.selectbox("Pilih Variabel Y", df_analysis.columns)
        
        if st.button("🔍 Hitung Korelasi"):
            valid_x, data_x = validate_numeric_data(df_analysis, var_x)
            valid_y, data_y = validate_numeric_data(df_analysis, var_y)
            
            if valid_x and valid_y:
                # Analisis
                r, p = pearsonr(data_x.dropna(), data_y.dropna())
                r_squared = r ** 2
                
                # Tampilkan hasil
                results_df = pd.DataFrame({
                    'Metrik': ['Koefisien Korelasi (r)', 'P-Value', 'R-squared'],
                    'Nilai': [r, p, r_squared]
                })
                st.table(results_df)
                
                # Interpretasi
                st.write("Interpretasi:")
                if p < 0.05:
                    if r > 0:
                        st.success(f"✅ Terdapat korelasi positif signifikan (r = {r:.3f})")
                    else:
                        st.success(f"✅ Terdapat korelasi negatif signifikan (r = {r:.3f})")
                else:
                    st.info("ℹ️ Tidak terdapat korelasi signifikan")
                
                # Visualisasi
                fig = px.scatter(df_analysis, x=var_x, y=var_y, trendline="ols")
                fig.update_layout(title=f"Korelasi antara {var_x} dan {var_y}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Kedua variabel harus berupa angka untuk analisis korelasi")
    
    elif analysis_type == "Marketing: ROI":
        col1, col2 = st.columns(2)
        with col1:
            biaya = st.number_input("Total Biaya (Rp)", min_value=0.0, format="%.2f")
        with col2:
            pendapatan = st.number_input("Total Pendapatan (Rp)", min_value=0.0, format="%.2f")
        
        if st.button("💰 Hitung ROI"):
            if biaya > 0:
                roi = ((pendapatan - biaya) / biaya) * 100
                profit = pendapatan - biaya
                
                # Tampilkan hasil
                results_df = pd.DataFrame({
                    'Metrik': ['ROI', 'Profit/Loss'],
                    'Nilai': [f"{roi:.2f}%", f"Rp {profit:,.2f}"]
                })
                st.table(results_df)
                
                # Visualisasi
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Biaya', 'Pendapatan', 'Profit/Loss'],
                    y=[biaya, pendapatan, profit],
                    text=[f"Rp {val:,.2f}" for val in [biaya, pendapatan, profit]],
                    textposition='auto',
                ))
                fig.update_layout(title="Analisis ROI")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Total biaya harus lebih dari 0")
    
    elif analysis_type == "Marketing: CTR":
        col1, col2 = st.columns(2)
        with col1:
            impressions = st.number_input("Jumlah Impressions", min_value=0)
        with col2:
            clicks = st.number_input("Jumlah Clicks", min_value=0)
        
        if st.button("🎯 Hitung CTR"):
            if impressions > 0:
                ctr = (clicks / impressions) * 100
                
                # Tampilkan hasil
                results_df = pd.DataFrame({
                    'Metrik': ['CTR', 'Impressions', 'Clicks'],
                    'Nilai': [f"{ctr:.2f}%", f"{impressions:,}", f"{clicks:,}"]
                })
                st.table(results_df)
                
                # Visualisasi
                fig = go.Figure(data=[go.Pie(
                    labels=['Clicks', 'No Action'],
                    values=[clicks, impressions-clicks],
                    hole=.3
                )])
                fig.update_layout(title="Click-through Rate Analysis")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Jumlah impressions harus lebih dari 0")

# Visualisasi Data Tambahan
if not df_analysis.empty:
    st.markdown("---")
    st.header("📈 Visualisasi Data Tambahan")
    
    viz_type = st.selectbox(
        "Pilih jenis visualisasi:",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Violin Plot", 
         "Heatmap Correlation", "Bubble Chart", "Area Chart"]
    )
    
    if viz_type != "Heatmap Correlation":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Pilih kolom untuk sumbu X", df_analysis.columns)
        with col2:
            y_col = st.selectbox("Pilih kolom untuk sumbu Y", df_analysis.columns)
        
        color_col = st.selectbox("Pilih kolom untuk pewarnaan (opsional)", 
                               ["Tidak ada"] + list(df_analysis.columns))
        
        if color_col == "Tidak ada":
            color_col = None
        
        try:
            if viz_type == "Bar Chart":
                fig = px.bar(df_analysis, x=x_col, y=y_col, color=color_col,
                           title=f"Bar Chart: {y_col} vs {x_col}")
            
            elif viz_type == "Line Chart":
                fig = px.line(df_analysis, x=x_col, y=y_col, color=color_col,
                            title=f"Line Chart: {y_col} vs {x_col}")
            
            elif viz_type == "Scatter Plot":
                fig = px.scatter(df_analysis, x=x_col, y=y_col, color=color_col,
                               title=f"Scatter Plot: {y_col} vs {x_col}")
            
            elif viz_type == "Box Plot":
                fig = px.box(df_analysis, x=x_col, y=y_col, color=color_col,
                           title=f"Box Plot: {y_col} by {x_col}")
            
            elif viz_type == "Violin Plot":
                fig = px.violin(df_analysis, x=x_col, y=y_col, color=color_col,
                              title=f"Violin Plot: {y_col} by {x_col}")
            
            elif viz_type == "Bubble Chart":
                size_col = st.selectbox("Pilih kolom untuk ukuran bubble", df_analysis.columns)
                fig = px.scatter(df_analysis, x=x_col, y=y_col, size=size_col,
                               color=color_col, title=f"Bubble Chart: {y_col} vs {x_col}")
            
            elif viz_type == "Area Chart":
                fig = px.area(df_analysis, x=x_col, y=y_col, color=color_col,
                            title=f"Area Chart: {y_col} vs {x_col}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error dalam pembuatan visualisasi: {str(e)}")
    
    else:  # Heatmap Correlation
        try:
            numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df_analysis[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix,
                              labels=dict(color="Correlation"),
                              title="Heatmap Correlation Matrix")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Minimal diperlukan 2 kolom numerik untuk membuat heatmap correlation")
        
        except Exception as e:
            st.error(f"❌ Error dalam pembuatan heatmap: {str(e)}")
