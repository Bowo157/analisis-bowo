import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
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
        # Coba konversi ke datetime
        try:
            if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0.7 * len(df):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            pass
        
        # Coba konversi ke numerik jika bukan datetime
        if df[col].dtype == 'object':
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.notna().sum() > 0.7 * len(df):  # Jika >70% bisa dikonversi
                    df[col] = numeric_col
            except Exception:
                pass
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
                        
                        # Ambil header dan data
                        header1 = raw_data[0]
                        header2 = raw_data[1]
                        data = raw_data[2:]
                        
                        # Deteksi kategori dari header pertama
                        categories = set()
                        for header in header1:
                            header_upper = str(header).upper()
                            if any(word in header_upper for word in ['BIOKIMIA', 'KINERJA', 'REPRODUKSI', 'DARAH', 'PRODUKTIF']):
                                categories.add(header)
                        
                        # Buat nama kolom final
                        new_columns = []
                        for i in range(len(header1)):
                            main_header = str(header1[i]).strip()
                            sub_header = str(header2[i]).strip()
                            
                            if main_header in categories:
                                col_name = sub_header
                            elif main_header == 'nan' or main_header == sub_header or not main_header:
                                col_name = sub_header
                            else:
                                col_name = main_header
                            
                            new_columns.append(col_name if col_name != 'nan' and col_name else f'Column_{i}')
                        
                        # Buat DataFrame dengan kolom yang benar
                        df = pd.DataFrame(data, columns=new_columns)
                        
                        # Konversi nilai numerik dengan hati-hati
                        for col in df.columns:
                            try:
                                # Skip kolom yang terlihat seperti ID, kategori, atau tanggal
                                if (col.upper() in ['NO', 'ID', 'BCS', 'KATEGORI'] or
                                    'TANGGAL' in col.upper() or 'TGL' in col.upper() or
                                    'WAKTU' in col.upper() or 'JAM' in col.upper() or
                                    'TIME' in col.upper() or 'DATE' in col.upper()):
                                    continue
                                    
                                # Bersihkan data
                                cleaned = df[col].str.strip()
                                # Cek apakah nilai terlihat seperti angka (tapi bukan tanggal)
                                is_numeric = cleaned.str.replace(',', '.').str.match(r'^\d*\.?\d+$')
                                # Skip jika terlihat seperti tanggal (angka dengan / atau -)
                                if cleaned.str.contains(r'\d+[/-]\d+').any():
                                    continue
                                
                                if is_numeric.any():
                                    # Konversi hanya nilai yang benar-benar numerik
                                    df.loc[is_numeric, col] = pd.to_numeric(
                                        cleaned[is_numeric].str.replace(',', '.'),
                                        errors='coerce'
                                    )
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

    # Form untuk menambah kolom
    with st.expander("➕ Tambah Kolom Baru"):
        col1, col2 = st.columns(2)
        with col1:
            nama_kolom = st.text_input("Nama Kolom:")
        with col2:
            tipe_data = st.selectbox(
                "Tipe Data:",
                ["number (angka)", "text (teks)", "date (tanggal)", "category (kategori)"]
            )
        
        if st.button("➕ Tambah Kolom"):
            if nama_kolom:
                if nama_kolom not in st.session_state.nama_kolom_manual:
                    st.session_state.nama_kolom_manual.append(nama_kolom)
                    st.session_state.tipe_data_kolom.append(tipe_data)
                    st.success(f"✅ Kolom {nama_kolom} ({tipe_data}) berhasil ditambahkan!")
                    st.session_state.data_manual = pd.DataFrame(
                        columns=st.session_state.nama_kolom_manual
                    )
                else:
                    st.error("❌ Nama kolom sudah ada!")
            else:
                st.error("❌ Nama kolom tidak boleh kosong!")

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
    
    # Tab untuk memilih jenis analisis
    analysis_type = st.radio(
        "Pilih Jenis Analisis:",
        ["Data Editor", "Statistik", "Marketing", "Visualisasi"],
        horizontal=True
    )

    if analysis_type == "Data Editor":
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
                        edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce')
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

    elif analysis_type == "Statistik":
        st.subheader("📊 Analisis Statistik")
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

        elif stat_type == "Uji Normalitas (Shapiro-Wilk)":
            st.write("Uji Shapiro-Wilk menguji apakah data berdistribusi normal.")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                var = st.selectbox("Pilih variabel untuk uji normalitas:", numeric_cols)
                
                try:
                    stat, p_val = shapiro(display_df[var].dropna())
                    
                    st.write("Hasil Uji Shapiro-Wilk:")
                    st.write(f"- Statistik uji: {stat:.4f}")
                    st.write(f"- P-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.info("Data tidak berdistribusi normal (p < 0.05)")
                    else:
                        st.success("Data berdistribusi normal (p > 0.05)")
                    
                    # Visualisasi histogram dan Q-Q plot
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = px.histogram(display_df, x=var, 
                                          title=f"Histogram: {var}",
                                          marginal="box")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        from scipy.stats import probplot
                        fig2 = go.Figure()
                        qq_data = probplot(display_df[var].dropna())
                        fig2.add_scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers')
                        fig2.add_scatter(x=qq_data[0][0], 
                                       y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                                       mode='lines')
                        fig2.update_layout(title="Q-Q Plot",
                                         xaxis_title="Theoretical Quantiles",
                                         yaxis_title="Sample Quantiles")
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Error dalam uji normalitas: {str(e)}")
            else:
                st.error("Minimal diperlukan 1 variabel numerik untuk uji normalitas")

        elif stat_type == "Uji Homogenitas (Levene)":
            st.write("Uji Levene menguji kesamaan varians antar grup.")
            
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
                        stat, p_val = levene(*groups)
                        
                        st.write("Hasil Uji Levene:")
                        st.write(f"- Statistik uji: {stat:.4f}")
                        st.write(f"- P-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.info("Varians antar grup berbeda secara signifikan (p < 0.05)")
                        else:
                            st.success("Varians antar grup homogen (p > 0.05)")
                        
                        # Visualisasi box plot
                        fig = px.box(display_df, x=group_var, y=dependent_var,
                                   title=f"Box Plot: {dependent_var} berdasarkan {group_var}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Minimal diperlukan 2 grup untuk uji homogenitas")
                except Exception as e:
                    st.error(f"Error dalam uji homogenitas: {str(e)}")
            else:
                st.error("Diperlukan minimal 1 variabel numerik dan 1 variabel kategorikal")

        elif stat_type == "Regresi Linear Sederhana":
            st.write("Regresi Linear Sederhana memprediksi variabel dependen berdasarkan satu variabel independen.")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Pilih variabel independen (X):", numeric_cols)
                with col2:
                    y_var = st.selectbox("Pilih variabel dependen (Y):", 
                                       [col for col in numeric_cols if col != x_var])
                
                try:
                    X = sm.add_constant(display_df[x_var])
                    model = sm.OLS(display_df[y_var], X).fit()
                    
                    st.write("Hasil Regresi Linear Sederhana:")
                    st.write(model.summary().tables[1])
                    
                    r_squared = model.rsquared
                    st.write(f"R-squared: {r_squared:.4f}")
                    
                    # Visualisasi scatter plot dengan garis regresi
                    fig = px.scatter(display_df, x=x_var, y=y_var,
                                   title=f"Scatter Plot dengan Garis Regresi: {y_var} vs {x_var}",
                                   trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error dalam analisis regresi: {str(e)}")
            else:
                st.error("Minimal diperlukan 2 variabel numerik untuk regresi linear sederhana")

        elif stat_type == "Regresi Linear Berganda":
            st.write("Regresi Linear Berganda memprediksi variabel dependen berdasarkan beberapa variabel independen.")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 3:
                y_var = st.selectbox("Pilih variabel dependen (Y):", numeric_cols)
                x_vars = st.multiselect(
                    "Pilih variabel independen (X):",
                    [col for col in numeric_cols if col != y_var],
                    default=[col for col in numeric_cols if col != y_var][:2]
                )
                
                if len(x_vars) >= 2:
                    try:
                        X = sm.add_constant(display_df[x_vars])
                        model = sm.OLS(display_df[y_var], X).fit()
                        
                        st.write("Hasil Regresi Linear Berganda:")
                        st.write(model.summary().tables[1])
                        
                        r_squared = model.rsquared
                        st.write(f"R-squared: {r_squared:.4f}")
                        
                        # Visualisasi prediksi vs aktual
                        predictions = model.predict(X)
                        fig = go.Figure()
                        fig.add_scatter(x=display_df[y_var], y=predictions,
                                      mode='markers',
                                      name='Prediksi vs Aktual')
                        fig.add_scatter(x=[display_df[y_var].min(), display_df[y_var].max()],
                                      y=[display_df[y_var].min(), display_df[y_var].max()],
                                      mode='lines',
                                      name='Perfect Prediction')
                        fig.update_layout(title="Prediksi vs Nilai Aktual",
                                        xaxis_title="Nilai Aktual",
                                        yaxis_title="Nilai Prediksi")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error dalam analisis regresi berganda: {str(e)}")
                else:
                    st.error("Pilih minimal 2 variabel independen")
            else:
                st.error("Minimal diperlukan 3 variabel numerik untuk regresi linear berganda")

        elif stat_type == "Uji Mann-Whitney U":
            st.write("Uji Mann-Whitney U membandingkan dua grup independen (alternatif non-parametrik dari t-test).")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = display_df.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    numeric_var = st.selectbox("Pilih variabel numerik:", numeric_cols)
                with col2:
                    cat_var = st.selectbox("Pilih variabel kategori (harus 2 grup):", categorical_cols)
                
                unique_groups = display_df[cat_var].unique()
                if len(unique_groups) == 2:
                    try:
                        group1 = display_df[display_df[cat_var] == unique_groups[0]][numeric_var]
                        group2 = display_df[display_df[cat_var] == unique_groups[1]][numeric_var]
                        
                        stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                        
                        st.write("Hasil Uji Mann-Whitney U:")
                        st.write(f"- Statistik U: {stat:.4f}")
                        st.write(f"- P-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success("Terdapat perbedaan signifikan antara grup (p < 0.05)")
                        else:
                            st.info("Tidak terdapat perbedaan signifikan antara grup (p > 0.05)")
                        
                        # Visualisasi box plot
                        fig = px.box(display_df, x=cat_var, y=numeric_var,
                                   title=f"Box Plot: {numeric_var} berdasarkan {cat_var}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error dalam uji Mann-Whitney U: {str(e)}")
                else:
                    st.error("Variabel kategori harus memiliki tepat 2 grup")
            else:
                st.error("Diperlukan minimal 1 variabel numerik dan 1 variabel kategorikal")

        elif stat_type == "Uji Wilcoxon":
            st.write("Uji Wilcoxon membandingkan dua pengukuran berpasangan (alternatif non-parametrik dari paired t-test).")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Pilih pengukuran pertama:", numeric_cols)
                with col2:
                    var2 = st.selectbox("Pilih pengukuran kedua:", 
                                      [col for col in numeric_cols if col != var1])
                
                try:
                    from scipy.stats import wilcoxon
                    stat, p_val = wilcoxon(display_df[var1], display_df[var2])
                    
                    st.write("Hasil Uji Wilcoxon:")
                    st.write(f"- Statistik W: {stat:.4f}")
                    st.write(f"- P-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.success("Terdapat perbedaan signifikan antara pengukuran (p < 0.05)")
                    else:
                        st.info("Tidak terdapat perbedaan signifikan antara pengukuran (p > 0.05)")
                    
                    # Visualisasi box plot
                    df_long = pd.melt(display_df[[var1, var2]])
                    fig = px.box(df_long, x='variable', y='value',
                               title=f"Box Plot: Perbandingan {var1} dan {var2}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error dalam uji Wilcoxon: {str(e)}")
            else:
                st.error("Minimal diperlukan 2 variabel numerik untuk uji Wilcoxon")

        elif stat_type == "Uji Kruskal-Wallis":
            st.write("Uji Kruskal-Wallis membandingkan tiga atau lebih grup independen (alternatif non-parametrik dari ANOVA).")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = display_df.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    numeric_var = st.selectbox("Pilih variabel numerik:", numeric_cols)
                with col2:
                    cat_var = st.selectbox("Pilih variabel kategori:", categorical_cols)
                
                unique_groups = display_df[cat_var].unique()
                if len(unique_groups) >= 3:
                    try:
                        from scipy.stats import kruskal
                        groups = [group for _, group in display_df.groupby(cat_var)[numeric_var]]
                        stat, p_val = kruskal(*groups)
                        
                        st.write("Hasil Uji Kruskal-Wallis:")
                        st.write(f"- Statistik H: {stat:.4f}")
                        st.write(f"- P-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success("Terdapat perbedaan signifikan antara grup (p < 0.05)")
                        else:
                            st.info("Tidak terdapat perbedaan signifikan antara grup (p > 0.05)")
                        
                        # Visualisasi box plot
                        fig = px.box(display_df, x=cat_var, y=numeric_var,
                                   title=f"Box Plot: {numeric_var} berdasarkan {cat_var}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error dalam uji Kruskal-Wallis: {str(e)}")
                else:
                    st.error("Variabel kategori harus memiliki minimal 3 grup")
            else:
                st.error("Diperlukan minimal 1 variabel numerik dan 1 variabel kategorikal")

        elif stat_type == "Uji Friedman":
            st.write("Uji Friedman membandingkan tiga atau lebih pengukuran berpasangan.")
            
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 3:
                selected_vars = st.multiselect(
                    "Pilih minimal 3 pengukuran:",
                    numeric_cols,
                    default=list(numeric_cols)[:3]
                )
                
                if len(selected_vars) >= 3:
                    try:
                        from scipy.stats import friedmanchisquare
                        stat, p_val = friedmanchisquare(*[display_df[var] for var in selected_vars])
                        
                        st.write("Hasil Uji Friedman:")
                        st.write(f"- Chi-square statistic: {stat:.4f}")
                        st.write(f"- P-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success("Terdapat perbedaan signifikan antara pengukuran (p < 0.05)")
                        else:
                            st.info("Tidak terdapat perbedaan signifikan antara pengukuran (p > 0.05)")
                        
                        # Visualisasi box plot
                        df_long = pd.melt(display_df[selected_vars])
                        fig = px.box(df_long, x='variable', y='value',
                                   title="Box Plot: Perbandingan Pengukuran")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error dalam uji Friedman: {str(e)}")
                else:
                    st.error("Pilih minimal 3 variabel untuk uji Friedman")
            else:
                st.error("Minimal diperlukan 3 variabel numerik untuk uji Friedman")

    elif analysis_type == "Marketing":
        st.subheader("📊 Analisis Marketing")
        st.write("Analisis metrik-metrik penting dalam marketing untuk mengukur efektivitas kampanye.")
        
        marketing_metric = st.selectbox(
            "Pilih Metrik Marketing:",
            ["ROI (Return on Investment)", 
             "CTR (Click Through Rate)", 
             "Conversion Rate",
             "CPA (Cost per Acquisition)",
             "CPC (Cost per Click)",
             "Impression Share",
             "CLV (Customer Lifetime Value)",
             "CAC (Customer Acquisition Cost)"]
        )

        if marketing_metric == "ROI (Return on Investment)":
            st.write("💰 Return on Investment (ROI)")
            st.write("ROI mengukur keuntungan atau kerugian yang dihasilkan dari investasi marketing.")
            
            col1, col2 = st.columns(2)
            with col1:
                revenue = st.number_input("Total Revenue (Pendapatan)", min_value=0.0, value=0.0)
                investment = st.number_input("Total Investment (Investasi)", min_value=0.0, value=0.0)
            
            if investment > 0:
                roi = ((revenue - investment) / investment) * 100
                st.metric("ROI", f"{roi:.2f}%")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Revenue', 'Investment'],
                    y=[revenue, investment],
                    text=[f'${revenue:,.2f}', f'${investment:,.2f}'],
                    textposition='auto',
                ))
                fig.update_layout(title="Revenue vs Investment")
                st.plotly_chart(fig, use_container_width=True)

        elif marketing_metric == "CTR (Click Through Rate)":
            st.write("🖱️ Click Through Rate (CTR)")
            st.write("CTR mengukur persentase orang yang mengklik iklan dari total yang melihat iklan.")
            
            col1, col2 = st.columns(2)
            with col1:
                clicks = st.number_input("Total Clicks", min_value=0, value=0)
                impressions = st.number_input("Total Impressions", min_value=0, value=0)
            
            if impressions > 0:
                ctr = (clicks / impressions) * 100
                st.metric("CTR", f"{ctr:.2f}%")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = ctr,
                    title = {'text': "Click Through Rate"},
                    gauge = {'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig, use_container_width=True)

        elif marketing_metric == "Conversion Rate":
            st.write("🎯 Conversion Rate")
            st.write("Conversion Rate mengukur persentase pengunjung yang melakukan tindakan yang diinginkan.")
            
            col1, col2 = st.columns(2)
            with col1:
                conversions = st.number_input("Total Conversions", min_value=0, value=0)
                visitors = st.number_input("Total Visitors", min_value=0, value=0)
            
            if visitors > 0:
                conv_rate = (conversions / visitors) * 100
                st.metric("Conversion Rate", f"{conv_rate:.2f}%")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = conv_rate,
                    title = {'text': "Conversion Rate"},
                    gauge = {'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig, use_container_width=True)

        elif marketing_metric == "CPA (Cost per Acquisition)":
            st.write("💵 Cost per Acquisition (CPA)")
            st.write("CPA mengukur biaya rata-rata untuk mendapatkan satu pelanggan.")
            
            col1, col2 = st.columns(2)
            with col1:
                total_cost = st.number_input("Total Marketing Cost", min_value=0.0, value=0.0)
                acquisitions = st.number_input("Total Acquisitions", min_value=0, value=0)
            
            if acquisitions > 0:
                cpa = total_cost / acquisitions
                st.metric("CPA", f"${cpa:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Cost per Acquisition'],
                    y=[cpa],
                    text=[f'${cpa:.2f}'],
                    textposition='auto',
                ))
                st.plotly_chart(fig, use_container_width=True)

        elif marketing_metric == "CPC (Cost per Click)":
            st.write("💰 Cost per Click (CPC)")
            st.write("CPC mengukur biaya rata-rata yang dibayarkan untuk setiap klik.")
            
            col1, col2 = st.columns(2)
            with col1:
                ad_cost = st.number_input("Total Ad Cost", min_value=0.0, value=0.0)
                total_clicks = st.number_input("Total Clicks", min_value=0, value=0)
            
            if total_clicks > 0:
                cpc = ad_cost / total_clicks
                st.metric("CPC", f"${cpc:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Cost per Click'],
                    y=[cpc],
                    text=[f'${cpc:.2f}'],
                    textposition='auto',
                ))
                st.plotly_chart(fig, use_container_width=True)

        elif marketing_metric == "Impression Share":
            st.write("👀 Impression Share")
            st.write("Impression Share mengukur persentase impresi yang diterima dari total impresi yang tersedia.")
            
            col1, col2 = st.columns(2)
            with col1:
                received = st.number_input("Received Impressions", min_value=0, value=0)
                available = st.number_input("Total Available Impressions", min_value=0, value=0)
            
            if available > 0:
                share = (received / available) * 100
                st.metric("Impression Share", f"{share:.2f}%")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = share,
                    title = {'text': "Impression Share"},
                    gauge = {'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig, use_container_width=True)

        elif marketing_metric == "CLV (Customer Lifetime Value)":
            st.write("💎 Customer Lifetime Value (CLV)")
            st.write("CLV memperkirakan total pendapatan yang diharapkan dari seorang pelanggan.")
            
            col1, col2 = st.columns(2)
            with col1:
                avg_purchase = st.number_input("Average Purchase Value", min_value=0.0, value=0.0)
                purchase_freq = st.number_input("Purchase Frequency (per year)", min_value=0.0, value=0.0)
                customer_lifespan = st.number_input("Customer Lifespan (years)", min_value=0.0, value=0.0)
            
            if all([avg_purchase, purchase_freq, customer_lifespan]):
                clv = avg_purchase * purchase_freq * customer_lifespan
                st.metric("CLV", f"${clv:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Average Purchase', 'Annual Value', 'Lifetime Value'],
                    y=[avg_purchase, avg_purchase * purchase_freq, clv],
                    text=[f'${v:.2f}' for v in [avg_purchase, avg_purchase * purchase_freq, clv]],
                    textposition='auto',
                ))
                fig.update_layout(title="CLV Components")
                st.plotly_chart(fig, use_container_width=True)

        elif marketing_metric == "CAC (Customer Acquisition Cost)":
            st.write("💰 Customer Acquisition Cost (CAC)")
            st.write("CAC mengukur biaya rata-rata untuk mendapatkan satu pelanggan baru.")
            
            col1, col2 = st.columns(2)
            with col1:
                marketing_costs = st.number_input("Total Marketing Costs", min_value=0.0, value=0.0)
                sales_costs = st.number_input("Total Sales Costs", min_value=0.0, value=0.0)
                new_customers = st.number_input("Number of New Customers", min_value=0, value=0)
            
            if new_customers > 0:
                total_costs = marketing_costs + sales_costs
                cac = total_costs / new_customers
                st.metric("CAC", f"${cac:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=['Marketing Costs', 'Sales Costs'],
                    values=[marketing_costs, sales_costs],
                    textinfo='label+percent'
                ))
                fig.update_layout(title="Cost Breakdown")
                st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Visualisasi":
        st.subheader("📈 Visualisasi Data")
        viz_type = st.selectbox(
            "Pilih Jenis Visualisasi:",
            ["Bar Chart - Perbandingan nilai antar kategori",
             "Line Chart - Tren waktu atau hubungan sekuensial",
             "Scatter Plot - Hubungan antara dua variabel numerik",
             "Histogram - Distribusi frekuensi variabel numerik",
             "Box Plot - Distribusi data dan outliers",
             "Pie Chart - Proporsi/komposisi kategori",
             "Heatmap - Korelasi antar variabel",
             "Area Chart - Area di bawah garis trend",
             "Bubble Chart - Scatter plot dengan variabel ukuran"]
        )
        
        # Pilih kolom untuk visualisasi berdasarkan tipe grafik
        if viz_type == "Heatmap - Korelasi antar variabel":
            create_visualization(display_df, viz_type)
        
        elif viz_type == "Histogram - Distribusi frekuensi variabel numerik":
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            x_col = st.selectbox("Pilih variabel untuk histogram:", numeric_cols)
            create_visualization(display_df, viz_type, x_col=x_col)
        
        elif viz_type in ["Bar Chart - Perbandingan nilai antar kategori", 
                         "Line Chart - Tren waktu atau hubungan sekuensial",
                         "Box Plot - Distribusi data dan outliers",
                         "Area Chart - Area di bawah garis trend"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("Pilih variabel X (kategori/waktu):", display_df.columns)
            with col2:
                numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
                y_col = st.selectbox("Pilih variabel Y (numerik):", numeric_cols)
            with col3:
                color_col = st.selectbox("Pilih variabel warna (opsional):", 
                                       ["Tidak ada"] + list(display_df.columns))
            color_col = None if color_col == "Tidak ada" else color_col
            create_visualization(display_df, viz_type, x_col, y_col, color_col)
        
        elif viz_type == "Pie Chart - Proporsi/komposisi kategori":
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Pilih variabel kategori:", 
                                     display_df.select_dtypes(include=['object']).columns)
            with col2:
                numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
                value_col = st.selectbox("Pilih variabel nilai:", numeric_cols)
            create_visualization(display_df, viz_type, x_col=cat_col, y_col=value_col)
        
        elif viz_type in ["Scatter Plot - Hubungan antara dua variabel numerik",
                         "Bubble Chart - Scatter plot dengan variabel ukuran"]:
            numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_col = st.selectbox("Pilih variabel X:", numeric_cols)
            with col2:
                y_col = st.selectbox("Pilih variabel Y:", 
                                   [col for col in numeric_cols if col != x_col])
            with col3:
                color_col = st.selectbox("Pilih variabel warna (opsional):", 
                                       ["Tidak ada"] + list(display_df.columns))
            with col4:
                size_col = st.selectbox("Pilih variabel ukuran (opsional):", 
                                      ["Tidak ada"] + list(numeric_cols))
            
            color_col = None if color_col == "Tidak ada" else color_col
            size_col = None if size_col == "Tidak ada" else size_col
            create_visualization(display_df, viz_type, x_col, y_col, color_col, size_col)

else:
    st.info("ℹ️ Silakan unggah atau input data terlebih dahulu untuk melihat analisis")
