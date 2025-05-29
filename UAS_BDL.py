import streamlit as st
import pandas as pd
from PIL import Image

st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
        }
        body, .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stSlider, .stDataFrame, .stTable {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True)

# Judul aplikasi
st.title("Visualisasi pada Big Data")

# Flowchart
st.header("Flowchart")
st.image("https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/Flowchart.jpg", caption="Diagram Flowchart")

# Data
st.header("Data: Sales of Supermarket")
data_url = "https://www.kaggle.com/datasets/lovishbansal123/sales-of-a-supermarket"
url = "https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/supermarket_sales.csv"
df = pd.read_csv(url)


# Potong data jadi 2000 baris
df = df.head(2000)

# Tampilkan data
st.dataframe(df)

# Caption & Markdown
st.caption(f"Sumber Data: [Kaggle - Sales of Supermarket]({data_url})")
st.markdown("""

Dataset ini berisi **1000 baris data transaksi penjualan** di sebuah supermarket yang beroperasi di tiga cabang berbeda. 
Data mencerminkan informasi lengkap mengenai aktivitas penjualan yang terjadi. Seperti tanggal, waktu, jenis produk, metode pembayaran, dan kepuasan pelanggan.
dengan ukuran data yang dimiliki berupa 1000 baris dan 17 kolom.

Variabel yang ada dalam dalam Dataset:
- `Invoice ID`: ID unik untuk setiap transaksi
- `Branch`: Cabang tempat transaksi dilakukan (A, B, atau C)
- `City`: Kota cabang (Yangon, Naypyitaw, Mandalay)
- `Customer type`: Jenis pelanggan (Member atau Normal)
- `Gender`: Jenis kelamin pelanggan
- `Product line`: Kategori produk yang dibeli (misalnya: Health and beauty, Food and beverages, dll.)
- `Unit price`: Harga per unit produk
- `Quantity`: Jumlah produk yang dibeli
- `Tax 5%`: Pajak penjualan sebesar 5%
- `Total`: Total harga setelah pajak
- `Date`: Tanggal transaksi
- `Time`: Waktu transaksi
- `Payment`: Metode pembayaran (Cash, Credit card, Ewallet)
- `COGS`: Biaya pokok penjualan
- `Gross margin percentage`: Persentase margin kotor
- `Gross income`: Pendapatan kotor
- `Rating`: Penilaian atau kepuasan pelanggan (skala 1–10).""")

# Feature Engineering
st.subheader("Feature Engineering")

selected_columns = ['Branch', 'City', 'Customer type', 'Gender', 
                    'Product line', 'Unit price', 'Quantity', 'Total', 
                    'Payment', 'Rating']
data_selected = df[selected_columns]

# Tampilkan data setelah seleksi variabel
st.dataframe(data_selected)

st.markdown("""
Variabel-variabel yang digunakan dalam analisis ini dipilih karena memiliki peran penting dalam menggambarkan pola dan tren penjualan di supermarket. Misalnya, variabel seperti Gender, Customer type, dan Payment berguna untuk memahami karakteristik serta perilaku pelanggan dalam segmentasi konsumen. Sementara itu, Unit price, Quantity, dan Total merupakan variabel kunci dalam menganalisis nilai transaksi. Variabel Product line memberikan gambaran preferensi pelanggan terhadap kategori produk tertentu, yang berguna untuk evaluasi produk dan strategi pemasaran. Branch dan City juga dipertahankan karena membantu melihat distribusi geografis transaksi dan kemungkinan perbedaan perilaku konsumen antar lokasi.

Sebaliknya, beberapa variabel tidak disertakan karena dinilai kurang relevan terhadap fokus analisis. contoh, Invoice ID hanya berupa identitas unik.

Pemilihan ini dilakukan untuk menjaga fokus, efisiensi, dan relevansi proses analisis terhadap tujuan yang ingin dicapai.
""")


st.subheader("Dashboard Visualisasi Data")
st.image("https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/visualisasi.png", caption="Visualisasi")
klik = "https://lookerstudio.google.com/reporting/05870969-5837-4da0-8423-261ea1d9643f"
st.caption(f"Sumber: [lookerstudio]({klik})")

st.subheader("Evaluation and Discussion")

st.markdown("""
Visualisasi data yang ditampilkan melalui Looker Studio memberikan gambaran yang jelas dan informatif mengenai pola penjualan di supermarket.
Dengan tampilan interaktif, kita mudah mengeksplorasi data berdasarkan lokasi cabang, jenis produk, hingga perilaku konsumen seperti metode pembayaran dan jenis pelanggan. 
Keunggulan utama dari Looker Studio adalah kemampuannya menyajikan data dalam bentuk grafik yang mudah dipahami dan mendukung analisis awal secara cepat.
selain kelebihan, terdapat keterbatasan seperti minimnya kemampuan untuk melakukan analisis lanjutan berbasis statistik atau prediktif.
Secara keseluruhan, visualisasi ini efektif dalam menyampaikan informasi penting dan mendukung pemahaman terhadap data penjualan. 
""")
