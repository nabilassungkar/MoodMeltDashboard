import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="MoodMelt's Media Intelligence Dashboard",
    page_icon="🍓",
    layout="wide"
)

# Colors for charts (Fruity theme: Tomato, Gold, LimeGreen, OrangeRed, Goldenrod, ForestGreen)
FRUITY_COLORS = ['#FF6347', '#FFD700', '#32CD32', '#FF4500', '#DAA520', '#228B22']

# --- Fungsi Bantuan ---

@st.cache_data
def parse_csv(uploaded_file):
    """
    Membaca dan membersihkan data dari file CSV yang diunggah.
    Mengonversi kolom 'Date' ke datetime dan mengisi 'Engagements' yang kosong.
    """
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        # Menggunakan io.StringIO untuk membaca file yang diunggah sebagai teks
        string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(string_data)

        # Membersihkan nama kolom dari spasi yang tidak diinginkan
        df.columns = df.columns.str.strip()

        # Pembersihan data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'Engagements' in df.columns:
            df['Engagements'] = pd.to_numeric(df['Engagements'], errors='coerce').fillna(0).astype(int)

        # Menghapus baris dengan tanggal atau keterlibatan yang tidak valid
        df.dropna(subset=['Date', 'Engagements'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error saat memproses file CSV: {e}")
        return pd.DataFrame()


def generate_campaign_summary(api_key, filtered_df):
    """
    Menghasilkan ringkasan kampanye menggunakan Gemini API.
    """
    if filtered_df.empty:
        return "Data tidak cukup untuk membuat ringkasan."

    # Agregasi data untuk prompt
    dominant_sentiment = filtered_df['Sentiment'].mode()[0] if not filtered_df['Sentiment'].empty else 'N/A'
    platform_engagement = filtered_df.groupby('Platform')['Engagements'].sum().sort_values(ascending=False)
    top_platform = platform_engagement.index[0] if not platform_engagement.empty else 'N/A'
    top_platform_engagements = int(platform_engagement.iloc[0]) if not platform_engagement.empty else 0

    engagement_trend = filtered_df.groupby('Date')['Engagements'].sum().sort_index()
    overall_trend = 'stabil'
    if len(engagement_trend) > 1:
        if engagement_trend.iloc[-1] > engagement_trend.iloc[0] * 1.1:
            overall_trend = 'meningkat'
        elif engagement_trend.iloc[-1] < engagement_trend.iloc[0] * 0.9:
            overall_trend = 'menurun'
    
    start_date = engagement_trend.index.min().strftime('%Y-%m-%d') if not engagement_trend.empty else 'N/A'
    end_date = engagement_trend.index.max().strftime('%Y-%m-%d') if not engagement_trend.empty else 'N/A'

    dominant_media_type = filtered_df['Media Type'].mode()[0] if 'Media Type' in filtered_df.columns and not filtered_df['Media Type'].empty else 'N/A'
    location_engagement = filtered_df.groupby('Location')['Engagements'].sum().sort_values(ascending=False)
    top_location = location_engagement.index[0] if not location_engagement.empty else 'N/A'
    top_location_engagements = int(location_engagement.iloc[0]) if not location_engagement.empty else 0

    prompt = f"""
    Berdasarkan data intelijen media dan wawasan berikut, berikan ringkasan strategi kampanye yang ringkas (tindakan dan rekomendasi utama).
    - Sentimen Dominan: {dominant_sentiment}.
    - Platform Keterlibatan Teratas: {top_platform} dengan {top_platform_engagements} keterlibatan.
    - Tren Keterlibatan Keseluruhan: {overall_trend} dari {start_date} hingga {end_date}.
    - Jenis Media yang Paling Sering Digunakan: {dominant_media_type}.
    - Lokasi Teratas untuk Keterlibatan: {top_location} dengan {top_location_engagements} keterlibatan.
    Sarankan 3-5 rekomendasi yang dapat ditindaklanjuti untuk mengoptimalkan kampanye media. Fokus pada langkah-langkah yang dapat ditindaklanjuti berdasarkan poin data ini.
    Format sebagai poin-poin.
    """
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status() # Akan error jika status code bukan 2xx
        result = response.json()
        
        if (result.get('candidates') and 
            result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts')):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Gagal membuat ringkasan. Respons API tidak terduga."
    except requests.exceptions.RequestException as e:
        return f"Error saat menghubungi API: {e}"
    except Exception as e:
        return f"Terjadi kesalahan: {e}"


# --- Tampilan UI ---

st.title("🍓 MoodMelt's Interactive Media Intelligence Dashboard")

# Bagian Unggah File
st.header("Unggah File CSV Anda 🍉")
uploaded_file = st.file_uploader(
    "Pastikan file Anda memiliki kolom 'Date', 'Engagements', 'Sentiment', 'Platform', 'Media Type', dan 'Location'.",
    type="csv"
)

# Menangani status jika tidak ada file yang diunggah
if uploaded_file is None:
    st.info("Silakan unggah file CSV untuk memulai analisis.")
    st.stop()

# Memproses data
df = parse_csv(uploaded_file)

if df.empty:
    st.warning("File CSV yang diunggah kosong atau tidak dapat diproses. Silakan periksa format file Anda.")
    st.stop()

# --- Sidebar Filter ---
st.sidebar.header("🥭 Filter Data")

# Filter
platform = st.sidebar.selectbox(
    "Pilih Platform",
    ['All'] + sorted(df['Platform'].unique().tolist())
)
sentiment = st.sidebar.selectbox(
    "Pilih Sentimen",
    ['All'] + sorted(df['Sentiment'].unique().tolist())
)
media_type = st.sidebar.selectbox(
    "Pilih Jenis Media",
    ['All'] + sorted(df['Media Type'].unique().tolist()) if 'Media Type' in df.columns else ['All']
)
location = st.sidebar.selectbox(
    "Pilih Lokasi",
    ['All'] + sorted(df['Location'].unique().tolist())
)

# Filter tanggal
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Menerapkan filter
filtered_df = df.copy()
if platform != 'All':
    filtered_df = filtered_df[filtered_df['Platform'] == platform]
if sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment]
if 'Media Type' in filtered_df.columns and media_type != 'All':
    filtered_df = filtered_df[filtered_df['Media Type'] == media_type]
if location != 'All':
    filtered_df = filtered_df[filtered_df['Location'] == location]

# Filter berdasarkan rentang tanggal
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

if filtered_df.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
    st.stop()


# --- Dasbor Utama ---

# Ringkasan Strategi Kampanye (AI)
st.subheader("🍒 Ringkasan Strategi Kampanye")
gemini_api_key = st.text_input("Masukkan Kunci API Gemini Anda", type="password")

if st.button("Buat Ringkasan Strategi"):
    if gemini_api_key:
        with st.spinner("Membuat ringkasan dengan Gemini..."):
            summary = generate_campaign_summary(gemini_api_key, filtered_df)
            st.markdown(summary)
    else:
        st.error("Silakan masukkan kunci API Gemini Anda untuk melanjutkan.")

# Layout kolom untuk grafik
col1, col2 = st.columns(2)

with col1:
    # Analisis Sentimen
    st.subheader("🍓 Analisis Sentimen")
    sentiment_data = filtered_df['Sentiment'].value_counts().reset_index()
    sentiment_data.columns = ['Sentiment', 'count']
    fig_sentiment = px.pie(
        sentiment_data,
        names='Sentiment',
        values='count',
        color_discrete_sequence=FRUITY_COLORS,
        hole=0.3
    )
    fig_sentiment.update_layout(legend_title_text='Sentimen')
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Keterlibatan per Platform
    st.subheader("🍋 Keterlibatan per Platform")
    platform_data = filtered_df.groupby('Platform')['Engagements'].sum().sort_values(ascending=False).reset_index()
    fig_platform = px.bar(
        platform_data,
        x='Platform',
        y='Engagements',
        color='Platform',
        color_discrete_sequence=FRUITY_COLORS,
        text_auto=True
    )
    fig_platform.update_layout(showlegend=False)
    st.plotly_chart(fig_platform, use_container_width=True)

with col2:
    # Kombinasi Jenis Media
    st.subheader("🥝 Kombinasi Jenis Media")
    if 'Media Type' in filtered_df.columns:
        media_type_data = filtered_df['Media Type'].value_counts().reset_index()
        media_type_data.columns = ['Media Type', 'count']
        fig_media_type = px.pie(
            media_type_data,
            names='Media Type',
            values='count',
            color_discrete_sequence=FRUITY_COLORS,
            hole=0.3
        )
        fig_media_type.update_layout(legend_title_text='Jenis Media')
        st.plotly_chart(fig_media_type, use_container_width=True)
    else:
        st.info("Kolom 'Media Type' tidak ditemukan di data.")
        
    # 5 Lokasi Teratas
    st.subheader("🍍 5 Lokasi Teratas")
    location_data = filtered_df.groupby('Location')['Engagements'].sum().nlargest(5).sort_values().reset_index()
    fig_location = px.bar(
        location_data,
        y='Location',
        x='Engagements',
        orientation='h',
        color='Location',
        color_discrete_sequence=FRUITY_COLORS,
        text='Engagements'
    )
    fig_location.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_location, use_container_width=True)


# Tren Keterlibatan Seiring Waktu
st.subheader("🍉 Tren Keterlibatan Seiring Waktu")
engagement_trend_data = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
fig_trend = px.line(
    engagement_trend_data,
    x='Date',
    y='Engagements',
    markers=True
)
fig_trend.update_traces(line=dict(color='#FF6347', width=3))
st.plotly_chart(fig_trend, use_container_width=True)

# Menampilkan data mentah yang difilter
if st.checkbox("Tampilkan data mentah yang difilter"):
    st.dataframe(filtered_df)
