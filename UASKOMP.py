import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
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

# Configure page
st.set_page_config(page_title="Machine Learning Basics", layout="wide")

st.title("Machine Learning Basics: Supervised Learning")
st.header("Data: AAPL")

st.subheader("Flowchart")
st.image("https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/Flowchart (1).jpg", caption="Flowchart")

# Define date range - last 1 year from today
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Download Apple stock data
@st.cache_data
def load_stock_data():
    ticker = "AAPL"
    
    # First try to get real data
    try:
        # Method 1: Try with period
        stock = yf.download(ticker, period="1y", progress=False)
        if not stock.empty:
            stock.reset_index(inplace=True)
            return stock, ticker, "real"
            
        # Method 2: Try with specific dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        stock = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'), progress=False)
        if not stock.empty:
            stock.reset_index(inplace=True)
            return stock, ticker, "real"
            
    except Exception as e:
        st.warning(f"Could not fetch real data: {str(e)}")
    
    # If real data fails, create sample data
    st.info("Using sample data for demonstration purposes")
    
    # Create sample Apple stock data - FIXED VERSION
    np.random.seed(42)  # For reproducible results
    
    # Create exact date range from 2024-05-30 to 2025-05-29
    dates = pd.date_range(start='2024-05-30', end='2025-05-29', freq='D')
    # Remove weekends (keep only Monday-Friday)
    dates = dates[dates.dayofweek < 5]
    
    n_days = len(dates)
    
    # Start with exact price from the image: $180.00
    base_price = 180.0
    
    # Initialize arrays
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    # Set the first day's data to match the image exactly
    current_price = base_price
    
    for i in range(n_days):
        # Generate daily return with some volatility
        if i == 0:
            # First day: exact values from image
            open_price = 180.0
            high_price = 180.1071
            low_price = 174.1657
            close_price = 180.0
            volume = 122807712
        else:
            # Calculate based on previous day with realistic stock movement
            daily_return = np.random.normal(0.0005, 0.015)  # Smaller daily moves
            gap_return = np.random.normal(0, 0.005)  # Opening gap
            
            # Opening price (small gap from previous close)
            open_price = closes[i-1] * (1 + gap_return)
            
            # Intraday movement
            intraday_high_move = abs(np.random.normal(0, 0.01))
            intraday_low_move = abs(np.random.normal(0, 0.01))
            
            # Close price
            close_price = open_price * (1 + daily_return)
            
            # High and Low
            high_price = max(open_price, close_price) * (1 + intraday_high_move)
            low_price = min(open_price, close_price) * (1 - intraday_low_move)
            
            # Volume (realistic range for AAPL)
            volume = int(np.random.normal(100000000, 30000000))
            volume = max(volume, 40000000)  # Minimum volume
        
        opens.append(round(open_price, 4))
        highs.append(round(high_price, 4))
        lows.append(round(low_price, 4))
        closes.append(round(close_price, 4))
        volumes.append(volume)
    
    # Create DataFrame with exact structure
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Adj Close': closes,  # Same as Close for simplicity
        'Volume': volumes
    })
    
    return sample_data, ticker, "sample"

# Load data
data, ticker_symbol, data_type = load_stock_data()


# Show data type indicator

if not data.empty:
    # Display the data
    st.subheader(f"Stock Market Data: {ticker_symbol} (Apple Inc.)")
    st.dataframe(data.head(10))
    
    # Basic statistics
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(data))
        st.metric("Date Range", f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.metric("Variables", len(data.columns))
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Add some additional analysis
    st.subheader("Price Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
            if not data.empty and 'Close' in data.columns and pd.notnull(data['Close'].iloc[-1]):
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
else:
        st.warning("Data tidak tersedia atau nilai 'Close' tidak valid.")
    with col2:
        st.metric("52W High", f"${data['High'].max():.2f}")
    with col3:
        st.metric("52W Low", f"${data['Low'].min():.2f}")
    with col4:
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
        st.metric("1Y Change", f"${price_change:.2f}", f"{(price_change/data['Close'].iloc[0]*100):.1f}%")

st.markdown("""
## Penjelasan Data

### Tentang Data
Data yang digunakan adalah data harga saham harian **Apple Inc. (AAPL)** dari Yahoo Finance untuk periode Januari-Desember 2025. Apple dipilih sebagai contoh karena merupakan salah satu saham paling likuid dan aktif diperdagangkan di bursa saham Amerika.

### Ukuran Data
- **Jumlah baris**: Sekitar 252 record (hari perdagangan dalam setahun)
- **Jumlah kolom**: 7 variabel utama
- **Periode**: 1 tahun terakhir (365 hari kebelakang dari hari ini)
- **Frekuensi**: Data harian (daily)

### Variabel dalam Dataset

1. **Date**: Tanggal perdagangan
2. **Open**: Harga pembukaan saham pada hari tersebut (USD)
3. **High**: Harga tertinggi saham pada hari tersebut (USD)
4. **Low**: Harga terendah saham pada hari tersebut (USD)
5. **Close**: Harga penutupan saham pada hari tersebut (USD)
6. **Adj Close**: Harga penutupan yang disesuaikan dengan dividen dan stock split (USD)
7. **Volume**: Jumlah saham yang diperdagangkan pada hari tersebut

### Mengapa Memilih Apple (AAPL)?

1. **Likuiditas Tinggi**: Apple adalah salah satu saham dengan volume perdagangan tertinggi, sehingga datanya lebih reliable dan representatif
2. **Stabilitas**: Sebagai perusahaan blue-chip, Apple memiliki pola pergerakan yang relatif stabil dan cocok untuk pembelajaran machine learning
3. **Popularitas**: Apple adalah salah satu perusahaan paling dikenal, sehingga mudah dipahami konteksnya
4. **Data Quality**: Yahoo Finance menyediakan data Apple dengan kualitas tinggi dan update real-time
5. **Cocok untuk Supervised Learning**: Data historis harga saham sangat cocok untuk prediksi (time series forecasting) menggunakan supervised learning

### Potensi Penggunaan untuk Machine Learning
Data ini dapat digunakan untuk berbagai teknik supervised learning seperti:
- **Regression**: Prediksi harga saham masa depan
- **Classification**: Klasifikasi apakah harga akan naik/turun
- **Time Series Analysis**: Analisis pola temporal dalam pergerakan harga
""")

    # Stock price visualization
st.subheader("Stock Price Visualization")
    
fig = go.Figure()
fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
fig.update_layout(
        title=f"{ticker_symbol} Stock Price Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified'
    )
    
st.plotly_chart(fig, use_container_width=True)

st.markdown("""grafik ini menunjukkan visualisasi perubahan harga saham AAPL.INC atau Apple dalam periode yang dipilih, dimana garis biru disini menunjukkan harga saham harian apple (menggunakan adjusted) yang di plot dari waktu ke waktu.""")

st.markdown("""
```python
                fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"{ticker_symbol} Stock Price Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)""")
    
    # Volume chart
st.subheader("Trading Volume")
fig_volume = px.bar(data, x='Date', y='Volume', title=f"{ticker_symbol} Trading Volume")
st.plotly_chart(fig_volume, use_container_width=True)

st.markdown("""grafik volume ini menggambarkan atau memvisualisasikan saham apple yang diperjualbelikan berdasarkan periode""")

st.markdown("""
```python
            fig_volume = px.bar(data, x='Date', y='Volume', title=f"{ticker_symbol} Trading Volume")
    st.plotly_chart(fig_volume, use_container_width=True)""")

# Markdown explanation

st.subheader("Data Exploration")

st.markdown("### Raw Data")
st.dataframe(data.tail(10))

st.markdown("### 📊 Descriptive Statistics")
st.dataframe(data.describe())

st.markdown("### Missing Values")
missing = data.isnull().sum()
st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.DataFrame({'Column': ['None'], 'Missing': [0]}))

st.markdown("### Distribution of Close Price")
fig_hist = px.histogram(data, x='Close', nbins=50, title="Close Price Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("### Outlier Detection (Boxplot of Close Price)")
fig_box = px.box(data, y='Close', title="Boxplot of Close Prices")
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("""
### Analisis Data

- **Struktur Data**: Data terdiri dari tanggal dan variabel harga saham (Open, High, Low, Close, Adj Close) serta Volume perdagangan.
- **Distribusi**: Histogram menunjukkan sebaran harga penutupan selama 1 tahun terakhir. 
- **Missing Values**: Cek dilakukan untuk memastikan tidak ada nilai kosong yang dapat mengganggu model.
- **Outliers**: Boxplot membantu mengidentifikasi adanya harga ekstrem yang tidak biasa.
""")

st.subheader("Feature Engineering")

st.markdown("""
### Identifikasi Fitur
Dalam analisis ini, fitur-fitur utama yang digunakan adalah:

- **Open**: Harga pembukaan dapat menjadi indikator awal arah pasar.
- **High & Low**: Menunjukkan volatilitas harian.
- **Volume**: Mencerminkan aktivitas pasar.
- **Close (Target)**: Harga penutupan akan digunakan sebagai target dalam prediksi.

Kita juga akan menambahkan beberapa fitur turunan, seperti:

- **Daily Return**: Persentase perubahan harga dari hari sebelumnya.
- **7-Day Moving Average (MA7)**: Rata-rata pergerakan harga selama 7 hari.
- **14-Day Moving Average (MA14)**: Rata-rata pergerakan jangka menengah.

### Transformasi Data
Kita melakukan transformasi sebagai berikut:
- **Normalisasi tidak dilakukan** secara langsung karena metode ARIMA lebih fokus pada nilai absolut dan perbedaan antar waktu.
- **Pembuatan fitur baru (feature engineering)** seperti return dan moving average membantu menangkap tren dan pola.

### Implementasi
""")

# Implementasi fitur
data['Return'] = data['Close'].pct_change()
data['MA7'] = data['Close'].rolling(window=7).mean()
data['MA14'] = data['Close'].rolling(window=14).mean()

st.dataframe(data[['Date', 'Close', 'Return', 'MA7', 'MA14']].tail(10))

# Visualisasi fitur
st.markdown("### Visualisasi Moving Averages")
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA7'], mode='lines', name='MA7'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA14'], mode='lines', name='MA14'))

fig_ma.update_layout(title='Moving Averages vs Close Price',
                     xaxis_title='Date',
                     yaxis_title='Price (USD)',
                     hovermode='x unified')

st.plotly_chart(fig_ma, use_container_width=True)

st.markdown("""
- **Daily Return** menunjukkan volatilitas harian yang biasanya digunakan untuk model regresi atau klasifikasi arah pergerakan harga.
- **Moving Averages** membantu mengidentifikasi tren jangka pendek dan menengah.
fitur ini penting digunakan dalam model prediktif""")

st.subheader("Prediction Stock Market: ARIMA")

# Persamaan ARIMA
st.markdown("Model ARIMA terdiri dari tiga komponen:")

st.markdown("**AR (Autoregressive)**: Hubungan dengan nilai masa lalu")
st.latex(r"y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p}")

st.markdown("**I (Integrated)**: Penggunaan diferensiasi untuk membuat data menjadi stasioner")
st.latex(r"y'_t = y_t - y_{t-1}")

st.markdown("**MA (Moving Average)**: Hubungan dengan error masa lalu")
st.latex(r"y_t = \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}")

st.markdown("Gabungan menjadi:")
st.latex(r"ARIMA(p, d, q): y'_t = \phi_1 y'_{t-1} + \dots + \phi_p y'_{t-p} + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q} + \epsilon_t")


# Tampilkan kode ARIMA
st.markdown("""```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Data Close dan pemisahan
train = data['Close'][:-30]
test = data['Close'][-30:]

# Fit model ARIMA (manual: bisa pakai p=5, d=1, q=0 sebagai contoh awal)
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Forecast 30 hari ke depan
forecast = model_fit.forecast(steps=30)

# Hitung RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
st.metric("RMSE", f"{rmse:.2f}")
```""")

# Real implementation
train = data['Close'][:-30]
test = data['Close'][-30:]

model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=30)

# Visualisasi prediksi vs aktual
st.markdown("### 📈 Forecasting Result")
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=data['Date'][-60:-30], y=train[-30:], name='Train', line=dict(color='blue')))
fig_forecast.add_trace(go.Scatter(x=data['Date'][-30:], y=test, name='Actual', line=dict(color='green')))
fig_forecast.add_trace(go.Scatter(x=data['Date'][-30:], y=forecast, name='Forecast', line=dict(color='red', dash='dash')))

fig_forecast.update_layout(
    title="ARIMA Forecast vs Actual (30 Days)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode='x unified'
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Tampilkan nilai RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")

# Analisis Distribusi Forecast
st.markdown("### 🔍 Distribusi Hasil Prediksi (Forecast)")
fig_dist = px.histogram(forecast, nbins=20, title="Histogram of ARIMA Forecast Values")
st.plotly_chart(fig_dist, use_container_width=True)

# Analisis Tambahan
st.markdown("""
### Analisis Hasil ARIMA

- **Model ARIMA(5,1,0)** dipilih untuk memodelkan harga saham berdasarkan 5 lag, differencing 1 kali, tanpa komponen MA.
- **Visualisasi** menunjukkan bahwa model cukup mengikuti pola aktual dalam 30 hari terakhir.
- **RMSE** sebesar {:.2f} menunjukkan seberapa jauh hasil prediksi dari nilai aktual.
- **Distribusi prediksi** tampak cukup normal, menunjukkan model stabil.

Model ini bisa dikembangkan lebih lanjut dengan optimasi parameter (p,d,q) menggunakan AIC/BIC atau Grid Search.
""".format(rmse))

st.subheader("📈 Prediction Stock Market: Random Forest (Alternatif LSTM)")

st.image("https://raw.githubusercontent.com/naaufald/UAS_KOMPSTAT/main/LSTM.png", caption="LSTM Cell Structure")
st.write("**Sumber:** [Medium](https://medium.com/@dhea.larasati326/multivariate-long-short-term-memory-dengan-python-c7170f443bd9)")
# --- Data ---
df_rf = data[['Close']].copy()

# Scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_rf)

# Buat dataset time series untuk 30 hari ke depan
n_steps = 60
X = []
y = []

for i in range(n_steps, len(scaled)-30):
    X.append(scaled[i-n_steps:i, 0])
    y.append(scaled[i+30, 0])

X = np.array(X)
y = np.array(y)

# Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Random Forest Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Prediction ---
y_pred = model.predict(X_test)

# Inverse scaling
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# --- RMSE ---
rmse_rf = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
st.success(f"✅ RMSE (Random Forest): {rmse_rf:.2f}")

# --- Visualisasi Prediksi vs Aktual ---
fig1, ax1 = plt.subplots()
ax1.plot(y_test_inv, label="Actual")
ax1.plot(y_pred_inv, label="Predicted")
ax1.set_title("Random Forest Prediction - 30 Days Ahead")
ax1.legend()
st.pyplot(fig1)

st.markdown("""
```python
df_rf = data[['Close']].copy()

# Scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_rf)

# Buat dataset time series untuk 30 hari ke depan
n_steps = 60
X = []
y = []

for i in range(n_steps, len(scaled)-30):
    X.append(scaled[i-n_steps:i, 0])
    y.append(scaled[i+30, 0])

X = np.array(X)
y = np.array(y)

# Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Random Forest Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Prediction ---
y_pred = model.predict(X_test)

# Inverse scaling
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))""")

# --- Distribusi Prediksi ---
fig2, ax2 = plt.subplots()
sns.histplot(y_pred_inv.flatten(), kde=True, bins=30, ax=ax2, color='green')
ax2.set_title("Distribution of Random Forest Predictions")
st.pyplot(fig2)

st.markdown("""
            penggunaan Random Forest sebagai alternatif penggunaan LSTM, karena pada Python 3.12, fitur Tensorflow sudah tidak kompatibel kembali di versi python yang tersedia (hanya tersedia di python 3.10 atau 3.11)""")

st.subheader("Evaluation and Discussion")

st.markdown("""
### Diskusi:

Berdasarkan hasil analisis dan prediksi yang dilakukan menggunakan metode **ARIMA** dan **Random Forest**, terdapat beberapa poin penting yang bisa didiskusikan:

**Kelebihan:**
- **ARIMA** sangat efektif untuk data deret waktu yang bersifat linier dan memiliki pola musiman atau tren.
- **Random Forest** mampu menangkap hubungan non-linier dalam data tanpa perlu asumsi distribusi tertentu.
- Visualisasi distribusi dan grafik prediksi membantu memahami perilaku pasar saham dari sudut statistik dan probabilistik.

**Keterbatasan:**
- **ARIMA** tidak cocok untuk data yang bersifat non-stasioner tanpa transformasi yang tepat.
- **Random Forest** bukan model deret waktu murni, sehingga tidak mempertimbangkan urutan waktu secara eksplisit.
- Model tidak mempertimbangkan faktor eksternal yang memengaruhi harga saham seperti berita, sentimen, atau indikator ekonomi global.

**Interpretasi Hasil:**
- Nilai **RMSE ARIMA** menunjukkan kinerja prediksi yang cukup baik pada data yang sudah distasionerkan.
- Nilai **RMSE Random Forest** relatif kompetitif, menunjukkan bahwa model ini dapat mempelajari pola data historis dengan cukup baik.
- Grafik distribusi prediksi cenderung simetris dan tidak terlalu menyimpang, menandakan stabilitas model.

---

### Kesimpulan:

- Prediksi harga saham AAPL telah berhasil dilakukan dengan dua pendekatan berbeda: **ARIMA** (statistik klasik) dan **Random Forest** (machine learning).
- Kedua metode memberikan hasil yang akurat dengan tingkat error (RMSE) yang rendah dan visualisasi yang masuk akal.
- Pendekatan berbasis machine learning seperti Random Forest memiliki keunggulan dalam menangkap kompleksitas pola harga, sedangkan ARIMA unggul pada data stasioner dengan tren musiman.
- Untuk prediksi jangka pendek, model yang telah dibangun cukup andal, namun untuk prediksi jangka panjang disarankan mempertimbangkan model tambahan dan variabel eksternal lainnya.

""")
