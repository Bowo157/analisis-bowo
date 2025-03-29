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
st.set_page_config(page_title="Analisis Ringan Bows", layout="wide")

# Judul aplikasi
st.title("📊 Analisis Data Ringan Bows Fakultas Peternakan")
st.markdown("Upload file Excel atau CSV kamu untuk melihat data dan analisis sederhana (bisa input data menggunakan 2 cara Excel atau Manual).")

# Inisialisasi session state jika belum ada
if 'nama_kolom_manual' not in st.session_state:
    st.session_state.nama_kolom_manual = []
if 'tipe_data_kolom' not in st.session_state:
    st.session_state.tipe_data_kolom = []
if 'data_manual' not in st.session_state:
    st.session_state.data_manual = pd.DataFrame()

# Tab untuk memilih metode input data
input_method = st.radio(
    "Pilih Metode Input Data:",
    ["Upload File", "Input Manual"],
    horizontal=True
)

if input_method == "Upload File":
    # Upload file dengan penanganan mobile-friendly
    st.markdown("""
    <style>
    /* CSS untuk membuat file uploader lebih mobile-friendly */
    .stFileUploader > div > div {
        padding: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 150px;
    }
    .stFileUploader > div > div > small {
        display: none;  /* Sembunyikan teks default */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Tambahkan petunjuk upload yang jelas
    st.markdown("""
    ### 📱 Petunjuk Upload File:
    1. Klik area upload di bawah
    2. Pilih "Browse" atau "Choose File"
    3. Pilih file dari perangkat Anda
    4. Format yang didukung: Excel (.xlsx, .xls) atau CSV (.csv)
    """)
    
    uploaded_file = st.file_uploader(
        "📁 Klik atau Tap di sini untuk memilih file", 
        type=['csv', 'xlsx', 'xls', 'ods', 'xlsb'],
        help="Pilih file Excel atau CSV dari perangkat Anda"
    )

    # Cek apakah ada file yang diunggah
    if uploaded_file is not None:
        with st.spinner('Membaca dan menganalisis file...'):
            try:
                # Cek tipe file dan baca isinya
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                # Baca file sesuai dengan tipe
                if file_type == 'csv':
                    try:
                        df = pd.read_csv(uploaded_file)
                    except UnicodeDecodeError:
                        df = pd.read_csv(uploaded_file, encoding='latin1')
                elif file_type in ['xlsx', 'xls', 'ods', 'xlsb']:
                    try:
                        # Baca Excel dengan openpyxl untuk kontrol penuh, tanpa konversi tanggal
                        workbook = openpyxl.load_workbook(uploaded_file, data_only=True)
                        sheet = workbook.active
                        
                        # Ambil data mentah dan pastikan semua nilai sebagai string
                        raw_data = []
                        for row in sheet.iter_rows(values_only=True):
                            row_data = []
                            for cell in row:
                                if cell is None:
                                    row_data.append('')
                                elif isinstance(cell, datetime):
                                    row_data.append(cell.strftime('%Y-%m-%d %H:%M:%S'))
                                else:
                                    row_data.append(str(cell))
                            raw_data.append(row_data)
                        
                        if len(raw_data) < 3:  # Minimal butuh 2 baris header + 1 baris data
                            st.error("❌ File Excel kosong atau tidak memiliki cukup data")
                            st.stop()
                        
                        # Baca header dan data dari Excel
                        header1 = [str(cell).strip() if cell else '' for cell in raw_data[0]]
                        header2 = [str(cell).strip() if cell else '' for cell in raw_data[1]]
                        data = raw_data[2:]
                        
                        # Gabungkan header jika ada 2 baris
                        new_columns = []
                        for i in range(len(header1)):
                            main_header = header1[i]
                            sub_header = header2[i] if i < len(header2) else ''
                            
                            # Jika header utama kosong, gunakan sub header
                            if not main_header or main_header.lower() == 'nan':
                                col_name = sub_header
                            # Jika sub header kosong atau sama dengan header utama, gunakan header utama
                            elif not sub_header or sub_header.lower() == 'nan' or sub_header == main_header:
                                col_name = main_header
                            # Jika keduanya ada dan berbeda, gabungkan
                            else:
                                col_name = f"{main_header} - {sub_header}"
                            
                            # Jika masih kosong, beri nama default
                            new_columns.append(col_name if col_name else f'Column_{i}')
                        
                        # Buat DataFrame dengan semua kolom sebagai string
                        df = pd.DataFrame(data, columns=new_columns, dtype=str)
                        
                        # Deteksi dan konversi kolom numerik
                        for col in df.columns:
                            try:
                                # Skip kolom yang berisi T0/T1
                                if df[col].str.contains('T0|T1').any():
                                    continue
                                
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
                                
                                # Jika bukan numerik dan bukan T0/T1, biarkan sebagai string
                                df[col] = cleaned
                            except:
                                continue
                        
                        # Hapus baris yang semuanya kosong
                        df = df.dropna(how='all')
                    except Exception as e:
                        st.error(f"❌ Error membaca file Excel: {str(e)}")
                        st.error("Pastikan format Excel sesuai dan tidak rusak")
                        st.stop()
                else:
                    st.error(f"❌ Format file .{file_type} tidak didukung")
                    st.stop()

                # Validasi data yang dibaca
                if df.empty:
                    st.error("❌ File tidak memiliki data")
                    st.stop()

                if len(df.columns) == 0:
                    st.error("❌ File tidak memiliki kolom yang valid")
                    st.stop()

                # Bersihkan dan preprocessing data
                df.columns = df.columns.str.strip()
                df = df.dropna(how='all')
                df = detect_column_types(df)

                # Simpan data ke session state
                st.session_state.data_manual = df
                st.session_state.nama_kolom_manual = list(df.columns)
                st.session_state.tipe_data_kolom = get_column_types(df)

                st.success("✅ File berhasil dimuat!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Pastikan file dalam format yang benar dan tidak rusak")
                st.session_state.data_manual = pd.DataFrame()
                st.session_state.nama_kolom_manual = []
                st.session_state.tipe_data_kolom = []

else:  # Input Manual
    st.markdown("---")
    st.header("📝 Input Data Manual")
    st.markdown("Input data secara manual dengan menentukan kolom dan tipe data.")

    # Form untuk membuat tabel
    with st.expander("📝 Buat Tabel Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            jumlah_data = st.number_input("Jumlah Data:", min_value=1, value=1)
        
        # Tampilkan form untuk mengisi nama dan tipe data
        if jumlah_data > 0:
            st.write("### Definisikan Data")
            kolom_baru = []
            tipe_data_baru = []
            
            # Buat input fields untuk setiap data
            for i in range(jumlah_data):
                col1, col2 = st.columns(2)
                with col1:
                    nama_kolom = st.text_input(f"Nama Data {i+1}:", key=f"nama_data_{i}")
                with col2:
                    tipe_data = st.selectbox(
                        f"Tipe Data {i+1}:",
                        ["number (angka)", "text (teks)", "date (tanggal)", "category (kategori)"],
                        key=f"tipe_data_{i}"
                    )
                if nama_kolom:
                    kolom_baru.append(nama_kolom)
                    tipe_data_baru.append(tipe_data)
            
            # Tombol untuk membuat tabel
            if st.button("✨ Buat Tabel") and len(kolom_baru) == jumlah_data:
                # Validasi nama data unik
                if len(set(kolom_baru)) != len(kolom_baru):
                    st.error("❌ Nama data harus unik!")
                elif "" in kolom_baru:
                    st.error("❌ Semua data harus diberi nama!")
                else:
                    st.session_state.nama_kolom_manual = kolom_baru
                    st.session_state.tipe_data_kolom = tipe_data_baru
                    st.session_state.data_manual = pd.DataFrame(columns=kolom_baru)
                    st.success("✅ Tabel berhasil dibuat!")
                    st.rerun()

    # Tampilkan tabel untuk input data jika struktur sudah dibuat
    if st.session_state.nama_kolom_manual:
        st.markdown("---")
        st.subheader("📝 Input Nilai Data")
        st.markdown("""
        ℹ️ **Petunjuk Penggunaan:**
        1. Gunakan tombol "➕ Tambah Baris" untuk menambah baris baru
        2. Klik pada sel untuk mengedit nilai
        3. Data akan otomatis tersimpan setelah diubah
        """)
        
        # Tambahkan tombol untuk menambah baris
        if st.button("➕ Tambah Baris"):
            new_row = pd.DataFrame([[None] * len(st.session_state.nama_kolom_manual)], 
                                 columns=st.session_state.nama_kolom_manual)
            st.session_state.data_manual = pd.concat([st.session_state.data_manual, new_row], 
                                                    ignore_index=True)
        
        # Tampilkan editor tabel dengan label yang lebih jelas
        st.markdown("##### Tabel Input Data:")
        edited_df = st.data_editor(
            st.session_state.data_manual,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            column_config={
                "_index": st.column_config.NumberColumn(
                    "No.",
                    help="Nomor urut data",
                    disabled=True
                )
            },
            key="manual_table_editor"
        )
        
        # Update data di session state
        st.session_state.data_manual = edited_df

    # Tombol untuk reset data
    if st.button("🗑️ Reset Data"):
        if st.session_state.nama_kolom_manual:
            st.session_state.nama_kolom_manual = []
            st.session_state.tipe_data_kolom = []
            st.session_state.data_manual = pd.DataFrame()
            st.success("✅ Data berhasil direset!")
            st.rerun()

# Inisialisasi DataFrame yang akan ditampilkan
display_df = st.session_state.data_manual.copy()

# Tampilkan tabel dengan opsi edit/hapus dan filter
if not display_df.empty:
    st.markdown("---")
    st.header("📊 Data dan Analisis")
    
    # Inisialisasi state untuk tracking langkah
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'data_edited' not in st.session_state:
        st.session_state.data_edited = False

    # Header untuk langkah analisis
    st.markdown("### 📊 Langkah Analisis Data")
    
    # Tampilkan langkah 1
    st.subheader("1️⃣ Input & Edit Data")
    
    st.markdown("---")
    
    # Step 1: Input & Edit Data
    if True:  # Always show step 1
        with st.expander("🔍 Filter dan Pengurutan Data", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter data
                filter_column = st.selectbox(
                    "Pilih kolom untuk filter:",
                    options=["Tidak ada filter"] + list(display_df.columns)
                )
                
                if filter_column != "Tidak ada filter":
                    if display_df[filter_column].dtype in ['int64', 'float64']:
                        display_df = filter_dataframe(display_df, filter_column, 'numeric')
                    elif display_df[filter_column].dtype == 'datetime64[ns]':
                        display_df = filter_dataframe(display_df, filter_column, 'datetime')
                    else:
                        display_df = filter_dataframe(display_df, filter_column, 'categorical')
            
            with col2:
                # Pengurutan data
                sort_column = st.selectbox(
                    "Urutkan berdasarkan:",
                    options=["Tidak ada pengurutan"] + list(display_df.columns)
                )
                
                if sort_column != "Tidak ada pengurutan":
                    display_df = sort_dataframe(display_df, sort_column)
            
            # Tampilkan informasi hasil filter
            st.info(f"📊 Menampilkan {len(display_df)} dari {len(st.session_state.data_manual)} baris data")

        # Tampilkan tabel dengan opsi edit
        edited_df = st.data_editor(
            display_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="data_editor"
        )

        # Update data jika ada perubahan
        if not edited_df.equals(st.session_state.data_manual):
            try:
                # Validasi tipe data
                for col, tipe in zip(st.session_state.nama_kolom_manual, st.session_state.tipe_data_kolom):
                    if tipe == "number (angka)":
                        edited_df[col] = pd.to_numeric(edited_df[col].astype(str).str.replace(',', '.'), errors='coerce')
                    elif tipe == "date (tanggal)":
                        edited_df[col] = pd.to_datetime(edited_df[col], errors='coerce')
                
                # Hapus baris kosong
                edited_df = edited_df.dropna(how='all')
                
                # Hitung perubahan
                rows_added = len(edited_df) - len(st.session_state.data_manual)
                changed_cells = (edited_df != st.session_state.data_manual).sum().sum()
                
                # Update data
                st.session_state.data_manual = edited_df
                
                # Tampilkan ringkasan perubahan
                with st.success("✅ Data berhasil diperbarui!"):
                    if rows_added > 0:
                        st.write(f"- {rows_added} baris baru ditambahkan")
                    elif rows_added < 0:
                        st.write(f"- {abs(rows_added)} baris dihapus")
                    if changed_cells > 0:
                        st.write(f"- {changed_cells} sel diubah")
                
            except Exception as e:
                st.error(f"❌ Error dalam memperbarui data: {str(e)}")
                st.error("Pastikan tipe data sesuai dengan yang ditentukan")

        # After data editing is done, enable step 2
        if not edited_df.equals(st.session_state.data_manual):
            st.session_state.data_edited = True
            
    # Step 2: Analisis Data
    if st.session_state.data_edited:
        st.markdown("---")
        st.subheader("2️⃣ Analisis Data")
        st.info("✨ Data telah siap untuk dianalisis!")
        stat_type = st.selectbox(
            "Pilih Jenis Analisis:",
            ["ANOVA (One-Way ANOVA)",
             "Korelasi Pearson",
             "Uji Chi-Square",
             "Uji Normalitas (Shapiro-Wilk)",
             "Uji Homogenitas (Levene)",
             "Regresi Linear Sederhana",
             "Regresi Linear Berganda",
             "Uji Mann-Whitney U",
             "Uji Wilcoxon",
             "Uji Kruskal-Wallis",
             "Uji Friedman"]
        )

        if stat_type == "ANOVA (One-Way ANOVA)":
            st.write("One-Way ANOVA membandingkan rata-rata antara tiga atau lebih grup.")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = display_df.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    dependent_var = st.selectbox("Pilih variabel dependen (numerik):", numeric_cols)
                with col2:
                    group_var = st.selectbox("Pilih variabel grup (kategori):", categorical_cols)
                
                try:
                    groups = [group for _, group in display_df.groupby(group_var)[dependent_var]]
                    if len(groups) >= 2:
                        f_stat, p_val = f_oneway(*groups)
                        
                        st.write("Hasil ANOVA:")
                        st.write(f"- F-statistic: {f_stat:.4f}")
                        st.write(f"- P-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success("Terdapat perbedaan signifikan antar grup (p < 0.05)")
                        else:
                            st.info("Tidak terdapat perbedaan signifikan antar grup (p > 0.05)")
                        
                        # Visualisasi box plot
                        fig = px.box(display_df, x=group_var, y=dependent_var,
                                   title=f"Box Plot: {dependent_var} berdasarkan {group_var}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Minimal diperlukan 2 grup untuk analisis ANOVA")
                except Exception as e:
                    st.error(f"Error dalam analisis ANOVA: {str(e)}")

        elif stat_type == "Korelasi Pearson":
            st.write("Korelasi Pearson mengukur kekuatan hubungan linear antara dua variabel numerik.")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Pilih variabel pertama:", numeric_cols)
                with col2:
                    var2 = st.selectbox("Pilih variabel kedua:", 
                                      [col for col in numeric_cols if col != var1])
                
                try:
                    correlation, p_value = pearsonr(
                        display_df[var1].dropna(),
                        display_df[var2].dropna()
                    )
                    
                    st.write("Hasil Korelasi Pearson:")
                    st.write(f"- Koefisien korelasi: {correlation:.4f}")
                    st.write(f"- P-value: {p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.success("Korelasi signifikan (p < 0.05)")
                    else:
                        st.info("Korelasi tidak signifikan (p > 0.05)")
                    
                    # Visualisasi scatter plot
                    fig = px.scatter(display_df, x=var1, y=var2, 
                                   title=f"Scatter Plot: {var1} vs {var2}",
                                   trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error dalam analisis korelasi: {str(e)}")
            else:
                st.error("Minimal diperlukan 2 variabel numerik untuk analisis korelasi")

        elif stat_type == "Uji Chi-Square":
            st.write("Uji Chi-Square menguji hubungan antara dua variabel kategorikal.")
            
            categorical_cols = display_df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Pilih variabel pertama:", categorical_cols)
                with col2:
                    var2 = st.selectbox("Pilih variabel kedua:", 
                                      [col for col in categorical_cols if col != var1])
                
                try:
                    contingency_table = pd.crosstab(display_df[var1], display_df[var2])
                    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                    
                    st.write("Hasil Uji Chi-Square:")
                    st.write(f"- Chi-square statistic: {chi2:.4f}")
                    st.write(f"- P-value: {p_val:.4f}")
                    st.write(f"- Degrees of freedom: {dof}")
                    
                    if p_val < 0.05:
                        st.success("Terdapat hubungan signifikan antara variabel (p < 0.05)")
                    else:
                        st.info("Tidak terdapat hubungan signifikan antara variabel (p > 0.05)")
                    
                    # Visualisasi heatmap
                    fig = px.imshow(contingency_table, 
                                  title="Heatmap Tabel Kontingensi",
                                  labels=dict(x=var2, y=var1, color="Frekuensi"))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error dalam uji Chi-Square: {str(e)}")
            else:
                st.error("Minimal diperlukan 2 variabel kategorikal untuk uji Chi-Square")

else:
    st.info("ℹ️ Silakan unggah atau input data terlebih dahulu untuk melihat analisis")
