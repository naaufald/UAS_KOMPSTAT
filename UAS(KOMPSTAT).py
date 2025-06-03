import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import math
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

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
    unsafe_allow_html=True
)

st.title("Machine Learning Basics: Supervised Learning")
st.header("Data: MSFT (Microsoft)")

st.subheader("Flowchart")
st.image("https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/Flowchart (1).jpg", caption="Flowchart")

msft = yf.Ticker("MSFT")
hist = msft.history(period = "1y")

data = pd.DataFrame(hist)
data

st.markdown("""
data ini merupakan data historis harga saham MSFT dari juni 2024 hingga 2 juni 2025. data yang ditampilkan terdiri dari 10 baris dan 8 kolom dengan masing-masing menunjukkan variabel dalam analisis pasar saham, seperti :
- Date: Tanggal perdagangan saham.
- Open: Harga pembukaan saham pada hari tersebut.
- High: Harga tertinggi yang dicapai saham selama hari perdagangan.
- Low: Harga terendah pada hari itu.
- Close: Harga penutupan saham pada akhir hari.
- Volume: Jumlah saham yang diperdagangkan.
- Dividends: Jumlah dividen yang dibayarkan pada hari itu (dalam dataset ini semuanya nol).
- Stock Splits: Informasi mengenai pemecahan saham (semuanya nol dalam data ini).""")

st.subheader("Data Exploration")

st.dataframe(data)

# Analisis struktur data
st.write("### Informasi Struktur Data")
st.write(data.info())

# Statistik deskriptif
st.write("### Statistik Deskriptif")
st.write(data.describe())

# Cek missing values
st.write("### Missing Values")
st.write(data.isnull().sum())

# Visualisasi Outliers (Boxplot)
st.write("### Boxplot Harga dan Volume")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=data[['Open', 'High', 'Low', 'Close', 'Volume']], ax=ax)
st.pyplot(fig)

st.markdown("""
```python
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=data[['Open', 'High', 'Low', 'Close', 'Volume']], ax=ax)
st.pyplot(fig)""")

# Visualisasi harga penutupan
st.write("### Harga Penutupan (Close Price) Saham MSFT")
fig_close = px.line(data, x=data.index, y='Close', title='Harga Penutupan Saham MSFT')
st.plotly_chart(fig_close)

st.markdown("""
```python
fig_close = px.line(data, x=data.index, y='Close', title='Harga Penutupan Saham MSFT')
st.plotly_chart(fig_close)""")

# Moving Average (MA20 dan MA50)
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

st.write("### Harga Penutupan dengan Moving Average (20 & 50 Hari)")
fig_ma = px.line(data, x=data.index, y=['Close', 'MA20', 'MA50'],
                 title='Harga Penutupan dengan MA20 dan MA50')
st.plotly_chart(fig_ma)

st.markdown("""
```python
fig_ma = px.line(data, x=data.index, y=['Close', 'MA20', 'MA50'],
                 title='Harga Penutupan dengan MA20 dan MA50')
st.plotly_chart(fig_ma)""")

# Volume perdagangan
st.write("### Volume Perdagangan Harian")
fig_volume = px.area(data, x=data.index, y='Volume', title='Volume Perdagangan Saham MSFT')
st.plotly_chart(fig_volume)

st.markdown("""
```python
fig_volume = px.area(data, x=data.index, y='Volume', title='Volume Perdagangan Saham MSFT')
st.plotly_chart(fig_volume)""")

st.subheader("Feature Engineering")

# Tampilkan kode dalam markdown
feature_code = '''
# Membuat fitur baru dari data historis
data['Daily Return'] = data['Close'].pct_change()
data['High-Low Spread'] = data['High'] - data['Low']
data['Open-Close Spread'] = data['Open'] - data['Close']
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Drop NaN hasil rolling dan return
data = data.dropna()
'''

st.markdown("```python\n" + feature_code + "\n```")

# Eksekusi kode feature engineering
data['Daily Return'] = data['Close'].pct_change()
data['High-Low Spread'] = data['High'] - data['Low']
data['Open-Close Spread'] = data['Open'] - data['Close']
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data = data.dropna()

# Tampilkan data hasil feature engineering
st.write("### Data Setelah Feature Engineering")
st.dataframe(data)

# Penjelasan fitur
st.write("### Penjelasan Feature Engineering")
st.markdown("""
- **Daily Return**: Persentase perubahan harga penutupan antar hari. Ini digunakan untuk melihat volatilitas dan tren harian.
- **High-Low Spread**: Mengukur volatilitas harian dengan menghitung selisih antara harga tertinggi dan terendah.
- **Open-Close Spread**: Mengukur pergerakan harga dalam satu hari perdagangan.
- **MA10 & MA50**: Moving Average 10 dan 50 hari digunakan untuk mengenali tren jangka pendek dan menengah.
- **Drop NA**: Setelah menggunakan rolling dan perubahan persentase, baris awal memiliki nilai kosong, maka dibuang agar model tidak error.
""")

st.subheader("Prediction Stock Market: ARIMA")

# === EQUATION ARIMA ===
st.markdown("### Persamaan Model ARIMA")
st.latex(r'''
ARIMA(p,d,q): \quad Y_t = c + \phi_1 Y_{t-1} + \ldots + \phi_p Y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \ldots + \theta_q \varepsilon_{t-q}
''')

# === DATA LOADING ===
msft = yf.Ticker("MSFT")
hist = msft.history(period="1y")
data = pd.DataFrame(hist)
close_data = data['Close']

# === ARIMA CODE ===
arima_code = '''
from statsmodels.tsa.arima.model import ARIMA

# Gunakan harga penutupan
train = close_data[:-30]
test = close_data[-30:]

# Fit model ARIMA (p=5, d=1, q=2)
model = ARIMA(train, order=(5, 1, 2))
model_fit = model.fit()

# Peramalan 30 hari ke depan
forecast = model_fit.forecast(steps=30)
'''
st.markdown("```python\n" + arima_code + "\n```")

# Train-test split
train = close_data[:-30]
test = close_data[-30:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 2))
model_fit = model.fit()

# Forecast 30 hari ke depan
forecast = model_fit.forecast(steps=30)

# === VISUALIZATION ===
st.write("### Visualisasi Prediksi Harga Saham 30 Hari ke Depan")
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Data Train'))
fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Data Test (Aktual)'))
fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast (ARIMA)', line=dict(dash='dash')))
st.plotly_chart(fig)

# === EVALUASI RMSE ===
rmse = math.sqrt(mean_squared_error(test, forecast))
st.write(f"### RMSE (Root Mean Squared Error): `{rmse:.4f}`")

# === ANALISIS SINGKAT ===
st.markdown("""
**Analisis**:
- ARIMA mampu mengikuti pola tren dengan cukup baik, namun terdapat sedikit penyimpangan pada fluktuasi harian yang tinggi.
- Distribusi prediksi mendekati distribusi aktual, meskipun sedikit lebih halus (karena model cenderung memprediksi rata-rata).
- RMSE menunjukkan tingkat kesalahan prediksi yang bisa digunakan untuk membandingkan model lain di masa depan.
""")

st.subheader("📈 Prediction Stock Market: Prophet")

st.image("https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/metode prophet.jpeg", caption="metode prophet")
st.write("Sumber: [medium](https://mochtarhdy24.medium.com/time-series-forecasting-dengan-fbprophet-python-5884a8cb2d2f)")

# === KODE PROPHET ===
prophet_code = '''
from prophet import Prophet

# Siapkan data untuk Prophet
df_prophet = data.reset_index()[['Date', 'Close']]
df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Inisialisasi dan fit model Prophet
model = Prophet()
model.fit(df_prophet)

# Buat dataframe tanggal untuk 30 hari ke depan
future = model.make_future_dataframe(periods=30)

# Lakukan prediksi
forecast = model.predict(future)
'''
st.markdown("```python\n" + prophet_code + "\n```")

# === PREPROCESS DATA ===
df_prophet = data.reset_index()[['Date', 'Close']]
df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True) 
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

# Fit model Prophet

model = Prophet()
model.fit(df_prophet) 

# Forecast 30 hari ke depan
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# === VISUALISASI PREDIKSI ===
st.write("### Visualisasi Prediksi Harga Saham 30 Hari ke Depan")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# === DISTRIBUSI DATA AKTUAL VS PREDIKSI ===
st.write("### Distribusi Harga Aktual vs Prediksi")
actual = df_prophet['y']
predicted = forecast['yhat'][:len(actual)]
fig2, ax2 = plt.subplots()
sns.kdeplot(actual, label='Actual', shade=True)
sns.kdeplot(predicted, label='Predicted', shade=True)
plt.legend()
st.pyplot(fig2)

# === EVALUASI RMSE ===
rmse = math.sqrt(mean_squared_error(actual, predicted))
st.write(f"### RMSE (Root Mean Squared Error): `{rmse:.4f}`")

# === ANALISIS SINGKAT ===
st.markdown("""
**Analisis**:
- Model Prophet berhasil mengidentifikasi tren jangka panjang dan melakukan peramalan untuk 30 hari ke depan.
- Distribusi hasil prediksi cukup mirip dengan data aktual, meskipun Prophet biasanya memprediksi dengan lebih halus.
- RMSE digunakan untuk mengukur ketepatan model; nilai yang rendah menunjukkan model cukup akurat.
""")

st.subheader("Untuk LSTM, sudah beberapa kali saya coba namun masih terkendala di library tensorflow maupun keras yang tidak kompatibel di versi python 3.12.5")

st.image("https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/LSTM.jpg", caption="LSTM")

st.markdown("""
Diskusi
### Kelebihan Analisis:
Exploratory Data Analysis (EDA) Komprehensif:

Telah dilakukan visualisasi boxplot, time series, dan moving average (MA10, MA20, MA50) yang memberikan gambaran yang jelas tentang tren harga saham MSFT selama satu tahun terakhir.

Fitur tambahan dari feature engineering seperti Daily Return, High-Low Spread, dan Open-Close Spread memberi wawasan lebih dalam tentang volatilitas harian dan tren pergerakan harga.

Model ARIMA:

ARIMA menunjukkan performa yang baik dalam menangkap tren historis jangka pendek.

Visualisasi prediksi vs aktual selama 30 hari terakhir menunjukkan kecenderungan prediksi yang mendekati data aktual.

Nilai RMSE yang dihitung memberikan metrik kuantitatif untuk menilai kesalahan prediksi.

Model Prophet:

Prophet dapat memodelkan tren dan musiman secara otomatis, dan cocok untuk data harian seperti harga saham.

Visualisasi dari prediksi dan distribusi hasil menunjukkan bahwa Prophet dapat mereplikasi bentuk umum tren harga saham dengan prediksi yang halus dan stabil.

Prophet juga memberikan interval ketidakpastian (confidence interval) yang berguna dalam memahami potensi rentang prediksi.

### Keterbatasan Analisis:
Tidak Memasukkan Faktor Eksternal:

- Baik ARIMA maupun Prophet hanya memanfaatkan data harga historis tanpa mempertimbangkan faktor fundamental maupun faktor eksternal

ARIMA Mengasumsikan Linearitas dan Stasioneritas:

- Karena harga saham bersifat non-stasioner dan memiliki dinamika kompleks, ARIMA bisa gagal menangkap perubahan mendadak yang dipengaruhi oleh faktor eksternal.

Prophet Cenderung Terlalu Halus (Smooth):

- Prophet cenderung tidak menangkap lonjakan mendadak (spike) atau penurunan tajam, sehingga hasil prediksi bisa tampak terlalu optimis atau konservatif.

### Kesimpulan
Berdasarkan analisis historis saham MSFT dari Juni 2024 hingga Juni 2025, baik ARIMA maupun Prophet mampu memberikan gambaran tren yang informatif.
Model ARIMA lebih cocok untuk pola jangka pendek yang tidak terlalu kompleks, sedangkan Prophet efektif untuk tren jangka panjang dan peramalan musiman.
Evaluasi menggunakan Root Mean Squared Error (RMSE) menunjukkan bahwa kedua model memiliki performa prediksi yang cukup baik, namun tetap perlu hati-hati karena volatilitas pasar tidak sepenuhnya dapat ditangkap oleh model statistik.
Kendala teknis pada LSTM membuka peluang untuk eksplorasi lebih lanjut setelah permasalahan kompatibilitas library diselesaikan.
Untuk keputusan investasi atau analisis pasar lebih dalam, disarankan untuk menggabungkan pendekatan kuantitatif ini dengan analisis fundamental dan sentimen pasar.""")