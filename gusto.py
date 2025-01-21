import streamlit as st # type: ignore
import pandas as pd # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import re
import plotly.graph_objects as go
import streamlit_dynamic_filters as DynamicFilters
from streamlit_dynamic_filters import DynamicFilters
from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from scipy.spatial.distance import cdist
from babel.numbers import format_currency
from datetime import datetime

import time

st.set_page_config(
    page_title='GUSTO DASHBOARD',          
    page_icon=None,           
    layout="wide",        
    initial_sidebar_state="auto"  
)
@st.cache_data
def read_data(file):
    df = pd.read_excel(file)
    return df

@st.cache_data
def read_data1(file):
    df = pd.read_excel(file)
    return df
try:
    dff = read_data('Gusto_my_record_18012025.xlsx')
    dp=read_data1('collection21012025withnovdec.xlsx')

except FileNotFoundError:
    st.error("The file  was not found. Please check the file path and try again.")
    st.stop()
current_date = datetime.now().date()
formatted_date = current_date.strftime("%d/%m/%y")
dp.columns = dp.columns.str.strip()
# st.write(dp.columns)
df2=pd.read_excel("erpidpending amount11022025withnovdec.xlsx")
df3=pd.read_excel("beat_total11022025withnovdec.xlsx")

dff["Google Maps Link"] = dff.apply(
    lambda row: f"https://www.google.com/maps?q={row['Latitude']},{row['Longitude']}",
    axis=1
)
dff["Last Modified On"] = pd.to_datetime(dff["Last Modified On"])

cutoff_date = datetime.now() - timedelta(days=9*30)

dff["Is Active"] = dff["Last Modified On"].apply(lambda x: "Active" if x >= cutoff_date else "Inactive")
dff['Particulars_number'] = dff['Outlet Erp Id'].str.extract(r'(\d{9})')

dff = dff[dff['Particulars_number'].notna()]
dff = dff.drop_duplicates(subset='Particulars_number', keep='first')

# dp['Particulars_number'] = dp['Particulars'].str.extract(r'(\d{9})')
dp['Particulars_number'] = dp['Particulars'].str.extract(r'(\d{9})')

dff['LAT'] = dff['Latitude'].astype(float)
dff['LONG'] = dff['Longitude'].astype(float)

dff.dropna(subset=['LAT', 'LONG'], inplace=True)

dp['Sum of Diff'] = pd.to_numeric(dp['Sum of Diff'], errors='coerce')

# Handle NaN values (optional: choose one approach below)
dp['Sum of Diff'] = dp['Sum of Diff'].fillna(0)
grouped_dp = dp.groupby('Particulars_number', as_index=False)['Sum of Diff'].sum()

#Rename the column to 'Total Pending'
grouped_dp.rename(columns={'Sum of Diff': 'Total Pending'}, inplace=True)

result = grouped_dp.merge(dff, on='Particulars_number', how='left')
sum_empty_pending = result.loc[result['MyBeat Plan'].isna(), 'Total Pending'].sum()

dff = dff.merge(grouped_dp, on='Particulars_number', how='left')

dff['Pending_Status'] = dff['Total Pending'].apply(
    lambda x: "No pending" if pd.isna(x) or x == 0 else "Pending"
)

dff = dff.merge(df2, on='Outlet Erp Id', how='left')

def process_dataframe(df):
    try:
        # Check if the required columns exist
        if 'Total Pending(11/01/25)' in df.columns and 'Total Pending' in df.columns:
            # Convert columns to numeric to avoid errors
            df['Total Pending(11/01/25)'] = pd.to_numeric(df['Total Pending(11/01/25)'], errors='coerce')
            df['Total Pending'] = pd.to_numeric(df['Total Pending'], errors='coerce')

            # Add the "Collected Amount" column (difference between the two columns)
            df['Collected Amount'] = df['Total Pending(11/01/25)'] - df['Total Pending']

            # Reorder columns to place 'Collected Amount' after 'Total Pending(11/01/25)'
            columns = df.columns.tolist()
            index = columns.index('Total Pending(11/01/25)')
            columns.insert(index + 1, columns.pop(columns.index('Collected Amount')))
            df = df[columns]

            # Calculate the percentage difference
            df.loc[:, 'Percent_diff'] = ((df['Total Pending(11/01/25)'] - df['Total Pending']) / df['Total Pending']) * 100
            df.loc[:, 'Percent_diff'] = df['Percent_diff'].apply(lambda x: 0 if abs(x) < 1e-10 else x)
            df.loc[:, 'Percent_diff'] = df['Percent_diff'].apply(lambda x: f"{x:.2f}%")
        else:
            st.error("Required columns 'Total Pending(11/01/25)' or 'Total Pending' are missing.")
    except Exception as e:
        st.error(f"An error occurred while processing the DataFrame: {e}")
    return df
dff = process_dataframe(dff)    
dd=dff
# st.write(dd)   
dynamic_filters = DynamicFilters(dff, filters=["Territory","Final_Beats","Is Active","Pending_Status"])    
dynamic_filters.display_filters(location='sidebar')
df = dynamic_filters.filter_df()


center_lat = np.mean(df['LAT'])
center_lon = np.mean(df['LONG'])
custom_color_map = {
    "Pending": "red",        
    "No pending": "green",    
      
}
fig = px.scatter_mapbox(
    df, 
    lat="LAT", 
    lon="LONG", 
    hover_name="Final_Beats",
     # Color the points based on the combined category
    hover_data=["Outlets Name","Zone","Region","Territory","Outlets Type"] ,
      # Color scheme for the combined categories
    zoom=10,  # Default zoom level; you can adjust this for a better view
    center={"lat": center_lat, "lon": center_lon},  # Center the map at a specific lat/lon
    height=600,  # Set the height of the map
    color="Pending_Status",
    # color_discrete_sequence=px.colors.qualitative.Set1,
    color_discrete_map=custom_color_map,    # Color the points based on the combined category
    
)


fig.update_traces(marker=dict(size=12, opacity=0.8))

# Set the map style (open street map is interactive with drag/zoom)
fig.update_layout(mapbox_style="open-street-map")

# Allow for pan and zoom on the map
fig.update_layout(
    mapbox=dict(
        center={"lat": center_lat, "lon": center_lon},
        zoom=10,  # Initial zoom level, user can zoom in/out
        accesstoken="your_mapbox_access_token"  # Optional: If using Mapbox's proprietary styles
    ),
    margin={"r":0,"t":0,"l":0,"b":0}  # Remove margins to ensure full map area
)

total_rows = dff.shape[0]
total_rows1=df.shape[0]


outlet_counts = df['Beats'].value_counts()
outlet_counts1 = dff['Final_Beats'].value_counts()
pending_outlets_count=df[df["Pending_Status"]=="Pending"].shape[0]

non_null_count = int(result['Total Pending'].sum())
nn=int(df["Total Pending(11/01/25)"].sum())
# st.write(nn)
formatted_amount5 = format_currency(nn, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
total_matched_amount=int(non_null_count-sum_empty_pending)
formatted_amount = format_currency(total_matched_amount, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
# formatted_amount = f"₹{total_matched_amount:,}".replace(',', ',')
n= int(df['Total Pending'].sum() )

formatted_amount3 = format_currency(n, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
dff = dff.drop_duplicates(subset="Outlets Name", keep="first")
new_df = dp[dp['Particulars_number'].isnull() | (dp['Particulars_number'] == '')]
new_df = new_df.reset_index(drop=True)
new_df["Particulars_number"] = new_df["Particulars"].map(
    dff.set_index("Outlets Name")["Particulars_number"]
)
col1, col2 ,col3 = st.columns(3, gap="small", vertical_alignment="top")
non_null_count1 = new_df['Sum of Diff'].sum()
r=int(non_null_count1+sum_empty_pending)
formatted_amount2 = format_currency(r, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")

with col1:
    st.metric(label=f"Matched records pending amount({formatted_date})", value=str(formatted_amount))
with col2:
    st.metric(label=f"unmatched records pending amount({formatted_date})", value=str(formatted_amount2))
with col3:
    st.metric(label="Universal outlets", value=f"{total_rows:,}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Beat outlets", value=f"{total_rows1:,}")
with col2:
    st.metric(label="Pending outlets", value=f"{pending_outlets_count:,}")
with col3:
    # st.metric(label="TOTAL pending amount / beat", value=str(formatted_amount4))
    st.metric(label="TOTAL Pending Amount / Beat(11/01/25)", value=str(formatted_amount5))
with col4:
    st.metric(label=f"Current Pending Amount / Beat({formatted_date})", value=str(formatted_amount3))    
    

st.plotly_chart(fig)

b= (
    dd.groupby(["Final_Beats"])["Total Pending"]
    .sum()
    .reset_index()
    # .rename(columns={"Total Pending": f"Total Pending({formatted_date})"})
)

selected_beats = st.multiselect(
    "Select Beat Names:",
    options=dff["Final_Beats"].unique(),  
    default=[]  
)
# dff = dff.merge(grouped_dp, on='Final_Beats', how='left')
d=df3.merge(b,on='Final_Beats', how='left')
d.index = range(1, len(d) + 1)  # Set index starting from 1
d.index.name = "Beat No"

d = process_dataframe(d)
d = process_dataframe(d)

d = d.rename(columns={"Total Pending": f"Total Pending({formatted_date})"})

if selected_beats: 
    fi = dff[dff["Final_Beats"].isin(selected_beats)]
    pending_outlets_df = fi[fi["Pending_Status"] == "Pending"]
    pending_outlets_df = pending_outlets_df[["Final_Beats", "Outlets Name","Total Pending(11/01/25)","Collected Amount" ,"Total Pending","Percent_diff"]]
    pending_outlets_df = pending_outlets_df.rename(columns={"Total Pending": f"Total Pending({formatted_date})"})
    pending_outlets_df = pending_outlets_df.sort_values(by="Final_Beats")
    pending_outlets_df.index = range(1, len(pending_outlets_df) + 1)  # Set index starting from 1
    pending_outlets_df.index.name = "SI No"
    st.write(pending_outlets_df)
else:
    st.write(d)
