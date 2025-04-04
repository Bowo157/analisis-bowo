import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import openpyxl
from scipy.stats import ttest_ind, f_oneway, pearsonr, shapiro, levene, mannwhitneyu, chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# Helper functions for data manipulation
def filter_dataframe(df, column, filter_type):
    """Helper function untuk memfilter dataframe"""
    try:
        if filter_type == 'numeric':
            min_val = float(df[column].min())
            max_val = float(df[column].max())
            filter_range = st.slider(
                f"Range nilai untuk {column}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            return df[
                (df[column] >= filter_range[0]) & 
                (df[column] <= filter_range[1])
            ]
        
        elif filter_type == 'datetime':
            min_date = df[column].min()
            max_date = df[column].max()
            date_range = st.date_input(
                f"Range tanggal untuk {column}:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                return df[
                    (df[column].dt.date >= date_range[0]) &
                    (df[column].dt.date <= date_range[1])
                ]
            return df
        
        elif filter_type == 'categorical':
            unique_values = df[column].unique()
            if len(unique_values) <= 10:
                selected_values = st.multiselect(
                    f"Pilih nilai untuk {column}:",
                    options=unique_values,
                    default=unique_values
                )
                if selected_values:
                    return df[df[column].isin(selected_values)]
                return df
            else:
                filter_value = st.text_input(
                    f"Filter nilai untuk {column}:",
                    help="Masukkan teks untuk mencari (case insensitive)"
                )
                if filter_value:
                    return df[
                        df[column].astype(str).str.contains(
                            filter_value, case=False, na=False
                        )
                    ]
                return df
        
        return df
    except Exception as e:
        st.error(f"❌ Error saat memfilter data: {str(e)}")
        return df

def sort_dataframe(df, column):
    """Helper function untuk mengurutkan dataframe"""
    try:
        sort_order = st.radio(
            "Urutan:",
            options=["Ascending ↑", "Descending ↓"],
            horizontal=True
        )
        
        # Cek tipe data kolom untuk pengurutan yang sesuai
        if df[column].dtype == 'datetime64[ns]':
            # Untuk data tanggal, urutkan berdasarkan timestamp
            return df.sort_values(
                by=column,
                ascending=(sort_order == "Ascending ↑"),
                na_position='last'
            )
        elif df[column].dtype in ['int64', 'float64']:
            # Untuk data numerik, urutkan sebagai angka
            return df.sort_values(
                by=column,
                ascending=(sort_order == "Ascending ↑"),
                na_position='last',
                key=lambda x: pd.to_numeric(x, errors='coerce')
            )
        else:
            # Untuk data teks, urutkan dengan mempertimbangkan case
            return df.sort_values(
                by=column,
                ascending=(sort_order == "Ascending ↑"),
                na_position='last',
                key=lambda x: x.str.lower() if x.dtype == 'object' else x
            )
    except Exception as e:
        st.warning(f"⚠️ Tidak dapat mengurutkan kolom {column}: {str(e)}")
        return df

def detect_column_types(df):
    """Deteksi tipe data kolom dan konversi jika memungkinkan"""
    for col in df.columns:
        try:
            # Skip kolom yang berisi T0/T1
            if df[col].dtype == 'object' and df[col].str.contains('T0|T1').any():
                continue
            
            if df[col].dtype == 'object':
                # Bersihkan data
                cleaned = df[col].str.strip()
                
                # Cek apakah kolom berisi angka dengan koma atau titik
                has_numbers = cleaned.str.replace(',', '.').str.match(r'^\d*\.?\d+$').any()
                
                if has_numbers:
                    # Konversi ke numerik dengan mengganti koma jadi titik
                    df[col] = pd.to_numeric(
                        cleaned.str.replace(',', '.'),
                        errors='coerce'
                    )
                    continue
                
                # Cek apakah kolom berisi format tanggal yang valid
                # Hanya jika tidak berisi angka murni
                if not has_numbers:
                    try:
                        datetime_col = pd.to_datetime(cleaned, errors='coerce')
                        if datetime_col.notna().sum() > 0.7 * len(df):
                            df[col] = datetime_col
                            continue
                    except:
                        pass
                
                # Jika bukan numerik dan bukan tanggal, biarkan sebagai string
                df[col] = cleaned
                
        except Exception:
            continue
            
    return df

def get_column_types(df):
    """Tentukan tipe data untuk setiap kolom"""
    tipe_data = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            tipe_data.append("number (angka)")
        elif df[col].dtype == 'datetime64[ns]':
            tipe_data.append("date (tanggal)")
        elif df[col].nunique() <= 10:
            tipe_data.append("category (kategori)")
        else:
            tipe_data.append("text (teks)")
    return tipe_data

def create_visualization(df, viz_type, x_col=None, y_col=None, color_col=None, size_col=None):
    """Helper function untuk membuat visualisasi"""
    try:
        if viz_type == "Bar Chart - Perbandingan nilai antar kategori":
            fig = px.bar(
                df, x=x_col, y=y_col, color=color_col,
                title=f"Bar Chart: {y_col} berdasarkan {x_col}",
                labels={x_col: x_col, y_col: y_col}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Bar Chart:**
            - Membandingkan nilai numerik antar kategori
            - Tinggi batang menunjukkan besaran nilai
            - Cocok untuk melihat perbedaan nilai antar grup
            """)
            
        elif viz_type == "Line Chart - Tren waktu atau hubungan sekuensial":
            fig = px.line(
                df, x=x_col, y=y_col, color=color_col,
                title=f"Line Chart: {y_col} terhadap {x_col}",
                labels={x_col: x_col, y_col: y_col}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Line Chart:**
            - Menunjukkan perubahan nilai sepanjang waktu/urutan
            - Garis menghubungkan titik-titik data secara berurutan
            - Cocok untuk analisis tren dan pola perubahan
            """)
            
        elif viz_type == "Scatter Plot - Hubungan antara dua variabel numerik":
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col, size=size_col,
                title=f"Scatter Plot: {y_col} vs {x_col}",
                labels={x_col: x_col, y_col: y_col}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Scatter Plot:**
            - Menunjukkan hubungan antara dua variabel numerik
            - Setiap titik mewakili satu observasi
            - Pola titik menunjukkan korelasi dan outliers
            """)
            
        elif viz_type == "Histogram - Distribusi frekuensi variabel numerik":
            fig = px.histogram(
                df, x=x_col,
                title=f"Histogram: Distribusi {x_col}",
                labels={x_col: x_col}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Histogram:**
            - Menunjukkan distribusi frekuensi data numerik
            - Tinggi batang menunjukkan frekuensi nilai
            - Cocok untuk melihat bentuk distribusi data
            """)
            
        elif viz_type == "Box Plot - Distribusi data dan outliers":
            fig = px.box(
                df, x=x_col, y=y_col,
                title=f"Box Plot: Distribusi {y_col}",
                labels={x_col: x_col, y_col: y_col}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Box Plot:**
            - Menunjukkan median, kuartil, dan outliers
            - Kotak menunjukkan IQR (Inter Quartile Range)
            - Titik di luar whisker adalah outliers
            """)
            
        elif viz_type == "Pie Chart - Proporsi/komposisi kategori":
            fig = px.pie(
                df, values=y_col, names=x_col,
                title=f"Pie Chart: Proporsi {x_col}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Pie Chart:**
            - Menunjukkan proporsi kategori dalam keseluruhan
            - Setiap bagian menunjukkan persentase
            - Cocok untuk data kategorikal dengan sedikit kategori
            """)
            
        elif viz_type == "Heatmap - Korelasi antar variabel":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Heatmap Korelasi",
                labels=dict(color="Korelasi")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Heatmap:**
            - Menunjukkan korelasi antar variabel numerik
            - Warna menunjukkan kekuatan dan arah korelasi
            - Cocok untuk melihat pola hubungan antar variabel
            """)
            
        elif viz_type == "Area Chart - Area di bawah garis trend":
            fig = px.area(
                df, x=x_col, y=y_col,
                title=f"Area Chart: {y_col} terhadap {x_col}",
                labels={x_col: x_col, y_col: y_col}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Area Chart:**
            - Seperti line chart dengan area terisi
            - Area menunjukkan besaran kumulatif
            - Cocok untuk data time series dan proporsi
            """)
            
        elif viz_type == "Bubble Chart - Scatter plot dengan variabel ukuran":
            fig = px.scatter(
                df, x=x_col, y=y_col,
                size=size_col, color=color_col,
                title=f"Bubble Chart: {y_col} vs {x_col}",
                labels={x_col: x_col, y_col: y_col}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Penjelasan Bubble Chart:**
            - Scatter plot dengan ukuran titik bervariasi
            - Ukuran bubble menunjukkan nilai variabel ketiga
            - Cocok untuk visualisasi 3 dimensi data
            """)
            
    except Exception as e:
        st.error(f"❌ Error dalam membuat visualisasi: {str(e)}")

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Data Bows", layout="wide")

# Judul aplikasi
st.title("📊 Analisis Data Bows Fakultas Peternakan")
st.markdown("Analisis data Anda dalam empat langkah mudah!")

# Inisialisasi session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'step' not in st.session_state:
    st.session_state.step = 1

def step_1():
    st.header("Langkah 1: Input Data")
    input_method = st.radio("Pilih Metode Input Data:", ["Upload File", "Input Manual"])

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Unggah file data Anda", type=['csv', 'xlsx', 'xls', 'ods', 'xlsb'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.data = df
                st.success("File berhasil diunggah!")
                st.session_state.step = 2
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
    else:
        st.markdown("### Input Data Manual")
        # [Keep the existing manual input code]

def step_2():
    st.header("Langkah 2: Pra-pemrosesan dan Eksplorasi Data")
    if st.session_state.data is not None:
        st.write(st.session_state.data.head())
        st.write(f"Jumlah baris: {len(st.session_state.data)}")
        st.write(f"Jumlah kolom: {len(st.session_state.data.columns)}")
        
        # Add options for data cleaning and preprocessing
        if st.checkbox("Hapus baris dengan nilai kosong"):
            st.session_state.data = st.session_state.data.dropna()
            st.success("Baris dengan nilai kosong telah dihapus.")
        
        if st.checkbox("Konversi tipe data kolom"):
            for col in st.session_state.data.columns:
                new_type = st.selectbox(f"Pilih tipe data untuk {col}", ["object", "int64", "float64", "datetime64"])
                try:
                    if new_type == "datetime64":
                        st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                    else:
                        st.session_state.data[col] = st.session_state.data[col].astype(new_type)
                except:
                    st.warning(f"Tidak dapat mengkonversi kolom {col} ke tipe {new_type}")
        
        if st.button("Lanjut ke Analisis"):
            st.session_state.step = 3

def step_3():
    st.header("Langkah 3: Analisis Data")
    analysis_type = st.selectbox("Pilih Jenis Analisis:", ["Statistik", "Marketing"])
    
    if analysis_type == "Statistik":
        stat_method = st.selectbox("Pilih Metode Statistik:", 
                                   ["Deskriptif", "ANOVA", "Korelasi", "Regresi", "Uji T", "Chi-Square"])
        if stat_method == "Deskriptif":
            st.write(st.session_state.data.describe())
        elif stat_method == "ANOVA":
            # [Keep existing ANOVA code]
        elif stat_method == "Korelasi":
            # [Keep existing Correlation code]
        # [Add other statistical methods]
    
    elif analysis_type == "Marketing":
        marketing_method = st.selectbox("Pilih Metode Marketing:", 
                                        ["Segmentasi Pelanggan", "Analisis RFM", "Analisis Keranjang Belanja"])
        if marketing_method == "Segmentasi Pelanggan":
            # Implement customer segmentation
            pass
        elif marketing_method == "Analisis RFM":
            # Implement RFM analysis
            pass
        elif marketing_method == "Analisis Keranjang Belanja":
            # Implement market basket analysis
            pass
    
    if st.button("Lanjut ke Visualisasi"):
        st.session_state.step = 4

def step_4():
    st.header("Langkah 4: Visualisasi Data")
    viz_type = st.selectbox("Pilih Jenis Visualisasi:", 
                            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", 
                             "Pie Chart", "Heatmap", "3D Scatter Plot"])
    
    if viz_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
        x_col = st.selectbox("Pilih kolom untuk sumbu X:", st.session_state.data.columns)
        y_col = st.selectbox("Pilih kolom untuk sumbu Y:", st.session_state.data.columns)
        
        if viz_type == "Bar Chart":
            fig = px.bar(st.session_state.data, x=x_col, y=y_col)
        elif viz_type == "Line Chart":
            fig = px.line(st.session_state.data, x=x_col, y=y_col)
        else:  # Scatter Plot
            fig = px.scatter(st.session_state.data, x=x_col, y=y_col)
    
    elif viz_type in ["Histogram", "Box Plot"]:
        col = st.selectbox("Pilih kolom:", st.session_state.data.columns)
        if viz_type == "Histogram":
            fig = px.histogram(st.session_state.data, x=col)
        else:  # Box Plot
            fig = px.box(st.session_state.data, y=col)
    
    elif viz_type == "Pie Chart":
        values_col = st.selectbox("Pilih kolom untuk nilai:", st.session_state.data.columns)
        names_col = st.selectbox("Pilih kolom untuk label:", st.session_state.data.columns)
        fig = px.pie(st.session_state.data, values=values_col, names=names_col)
    
    elif viz_type == "Heatmap":
        fig = px.imshow(st.session_state.data.corr())
    
    elif viz_type == "3D Scatter Plot":
        x_col = st.selectbox("Pilih kolom untuk sumbu X:", st.session_state.data.columns)
        y_col = st.selectbox("Pilih kolom untuk sumbu Y:", st.session_state.data.columns)
        z_col = st.selectbox("Pilih kolom untuk sumbu Z:", st.session_state.data.columns)
        fig = px.scatter_3d(st.session_state.data, x=x_col, y=y_col, z=z_col)
    
    st.plotly_chart(fig)

# Main app logic
if st.session_state.step == 1:
    step_1()
elif st.session_state.step == 2:
    step_2()
elif st.session_state.step == 3:
    step_3()
elif st.session_state.step == 4:
    step_4()

# Navigation buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.step > 1:
        if st.button("⬅️ Kembali"):
            st.session_state.step -= 1
            st.experimental_rerun()
with col3:
    if st.session_state.step < 4:
        if st.button("Lanjut ➡️"):
            st.session_state.step += 1
            st.experimental_rerun()
