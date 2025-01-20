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

# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
# import folium
# from folium.plugins import MarkerCluster
# from sklearn.cluster import DBSCAN
# from geopy.distance import geodesic
# from geopy.geocoders import Nominatim
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
    dp=read_data1('colection18012025.xlsx')

except FileNotFoundError:
    st.error("The file  was not found. Please check the file path and try again.")
    st.stop()

dp.columns = dp.columns.str.strip()
df2=pd.read_excel("erpid_prnding110125.xlsx")
df3=pd.read_excel("Gusto_beat_count.xlsx")
# st.write(df3)
# df4=pd.read_excel("PMUK For LocationUpdationNew (1).xlsx")

# df4= df4.rename(columns={"Shop ERP Id": "Outlet Erp Id"})
# dff = dff.merge(df4[['Outlet Erp Id', 'GUID']], on='Outlet Erp Id', how='left')

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
# st.write(dp)
# empty_df = dp[pd.isna(dp['Particulars_number'])]

# # Keep rows with valid Particulars_number in the original DataFrame
# dp = dp[~pd.isna(dp['Particulars_number'])]
# Step 2: For remaining None values, extract 8 digits
# dp.loc[dp['Particulars_number'].isnull(), 'Particulars_number'] = dp['Particulars'].str.extract(r'(\d{8})')

# Step 3: For remaining None values, map ERP ID using names
# Remove duplicates in dff based on "Outlets Name"
df = dff.drop_duplicates(subset="Outlets Name", keep="first")

# # Map remaining None values based on names
# dp.loc[dp['Particulars_number'].isnull(), 'Particulars_number'] = dp.loc[dp['Particulars_number'].isnull(), 'Particulars'] \
#     .map(df.set_index("Outlets Name")["Particulars_number"])

dff['LAT'] = dff['Latitude'].astype(float)
dff['LONG'] = dff['Longitude'].astype(float)

dff.dropna(subset=['LAT', 'LONG'], inplace=True)

# n_clusters = int(np.ceil(len(dff) / 60)) 
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# dff['Cluster'] = kmeans.fit_predict(dff[['Latitude', 'Longitude']])
# dff['MyBeat Plan'] = 'Beat' + (dff['Cluster'] + 1).astype(str) 
# beat_counts = dff['MyBeat Plan'].value_counts()

# large_beats = beat_counts[beat_counts > 65].index.tolist()
# for beat in large_beats:
#     # Filter rows belonging to the large beat
#     subset = dff[dff['MyBeat Plan'] == beat].copy()
#     count = len(subset)

#     # Step 3: Determine how many sub-clusters to create
#     if  65 < count <= 140:
#         n_sub_clusters = 2  # Split into 2 clusters
#     elif count > 140:
#         n_sub_clusters = 3  # Split into 3 clusters
#     else:
#         continue  # No need to split if the count is <= 80

#     # Step 4: Sort the subset by latitude and longitude (if necessary)
#     subset = subset.sort_values(by=['Latitude', 'Longitude'])

#     # Step 5: Split the subset into `n_sub_clusters` equal parts
#     # If n_sub_clusters = 2, split the subset into two halves, if n_sub_clusters = 3, split into thirds
#     split_size = int(np.ceil(count / n_sub_clusters))
#     subclusters = [subset.iloc[i:i + split_size] for i in range(0, count, split_size)]

#     # Step 6: Assign new labels to the sub-clusters
#     for i, subcluster in enumerate(subclusters):
#         # Ensure you are assigning the value back to the correct column using `.loc`
#         dff.loc[subcluster.index, 'MyBeat Plan'] = subcluster['MyBeat Plan'] + chr(65 + i)


dp['Sum of Diff'] = pd.to_numeric(dp['Sum of Diff'], errors='coerce')

# Handle NaN values (optional: choose one approach below)
dp['Sum of Diff'] = dp['Sum of Diff'].fillna(0)
grouped_dp = dp.groupby('Particulars_number', as_index=False)['Sum of Diff'].sum()

# Step 3: Rename the column to 'Total Pending'
grouped_dp.rename(columns={'Sum of Diff': 'Total Pending'}, inplace=True)

result = grouped_dp.merge(dff, on='Particulars_number', how='left')
sum_empty_pending = result.loc[result['MyBeat Plan'].isna(), 'Total Pending'].sum()
dff = dff.merge(grouped_dp, on='Particulars_number', how='left')

dff['Pending_Status'] = dff['Total Pending'].apply(lambda x: "No pending" if pd.isna(x) else "Pending")
# beat_to_area_map = {
#     "Beat42":"Channasandra",
#     "Beat29":"Laggere",
#     "Beat20A":"Uttarahalli",
#     "Beat20B":"Kumaraswamy layout",
#     "Beat62":"Jalahalli",
#     "Beat34":"Nagarabavi",
#     "Beat61":"Basaweshwara nagara",
#     "Beat41":"Kaggadasapura",
#     "Beat45":"Mallathahalli",
#     "Beat46A":"RT nagara",
#     "Beat46B":"Hebbal",
#     "Beat25B":"Hennur",
#     "Beat25A":"Kammanahalli",
#     "Beat14C":"Taverekere",
#     "Beat14B":"Madivala",
#     "Beat14A":"BTM Layout",
#     "Beat7B":"Dasarahalli",
#     "Beat7A":"Peenya",
#     "Beat38A":"Koramangala",
#     "Beat38B":"Ejipura",
#     "Beat66":"Sunkadakatte",
#     "Beat4A":"Ramamurthy nagara",
#     "Beat4B":"Horamavu",
#     "Beat72B":"Bagalaguntte-2",
#     "Beat72A":"Bagalaguntte-1",
#     "Beat43B":"Chamarajapete",
#     "Beat43A" :"Banashankari",
#     "Beat23A":"JP NAGARA 1",
#     "Beat23B":"JP NAGARA 2",
#     "Beat69A":"Rajajinagara",
#     "Beat69B":"Mahalakshmi layout",
#     "Beat3B":"Haralur",
#     "Beat47":"Haralur",
#     "Beat3A":"Kudlu gate",
#     "Beat67":"TC Palya",
#     "Beat52":"Electronic city",
#     "Beat17":"Electronic city",
#     "Beat32A":"Sanjaya nagara",
#     "Beat32B":"Nagashetty halli",
#     "Beat9":"R K Hegde nagara",
#     "Beat54":"Ullalu",
#     "Beat18":"Andralli",
#     "Beat73B":"Hosakerehalli",
#     "Beat73A":"Padmanabhanagara",
#     "Beat59":"Arekere",
#     "Beat22B":"Mattikere",
#     "Beat22A":"Yashwanthpura",
#     "Beat16":"Indiranagara",
#     "Beat58":"Yelahanka",
#     "Beat68":"BEL",
#     "Beat35B":"R R Nagara 1",
#     "Beat35A":"R R Nagara 2",
#     "Beat24B":"Jakkur",
#     "Beat24A":"Kemmapura",
#     "Beat51A":"Shivaji nagara",
#     "Beat51B":"Frazer Town",
#     "Beat55B":"Ashwath nagara",
#     "Beat55A":"HBR Layout",
#     "Beat48":"yelahanka new town",
#     "Beat21":"Bagalur",
#     "Beat60":"Kothanur",
#     "Beat74":"V V Puram",
#     "Beat63A":"HSR Layout",
#     "Beat63B":"Sarajpuara Road",
#     "Beat5":"MS Palya",
#     "Beat49A":"Hulimavu",
#     "Beat49B":"Hulimavu",
#     "Beat1A":"Vijayanagara",
#     "Beat1B":"Vijayanagara",
#     "Beat27A":"Seshadripuram",
#     "Beat27B":"Seshadripuram",
#     "Beat6A":"Jayanagara/BSK-1",
#     "Beat6B":"Jayanagara/BSK-2",
#     "Beat65B":"Jayanagara",
#     "Beat65A":"Jayanagara",
#     "Beat33":"whitefeild",
#     "Beat13":"marathahalli",
#     "Beat11":"Kengere",
#     "Beat64":"Kengere",
#     "Beat8A":"Ganganagara",
#     "Beat8B":"Ganganagara",
#     "Beat39B":"Vidyaranyapura",
#     "Beat39A":"Vidyaranyapura",
#     "Beat50A":"Chikkabidarakallu",
#     "Beat50B":"Chikkabidarakallu",
#     "Beat37":"Nelamangala",
#     "Beat10":"Nelamangala",
#     "Beat26":"Hosakote",
#     "Beat15":"Chikka banavara",
#     "Beat57":"Hesaragatta",
#     "Beat44":"Hesaragatta",
#     "Beat56":"Anjana nagara",
#     "Beat70":"Vittasandra",
#     "Beat53":"Devanahalli",
#     "Beat19":"Bagalur",
#     "Beat36":"Ambedkar nagara"
#     # Add more mappings here as needed
# }  
# dff['MyBeat Plan'] = dff['MyBeat Plan'].astype(str)
# dff['final_beat_plan'] = dff['MyBeat Plan'].map(beat_to_area_map)

# Handle NaN values that might arise from unmatched beats
# dff['final_beat_plan'] = dff['final_beat_plan'].fillna('Mysore')
  
# dff = dff.merge(df2, on='final_beat_plan', how='left')
# dff.loc[dff['final_beat_plan'].isin(["Bagalaguntte-1", "Bagalaguntte-2"]), 'TSI NAME'] = "NARASIMHARAJU"
# dff.loc[dff['final_beat_plan'].isin(["Electronic city","Vittasandra"]), 'TSI NAME'] = "ABHISHEK"

# dff= dff.rename(columns={"final_beat_plan": "Final_Beats"})
# dff.loc[dff['Final_Beats'].isin(["whitefeild", "marathahalli"]), 'TSI NAME'] = 'ABHISHEK'
# dff.loc[dff['Final_Beats']=="Banashankari", 'TSI NAME'] = 'RAMESH'
# dff['Position'] = ''
# dff.loc[dff['TSI NAME'] == 'ABHISHEK', 'Position'] = 'Kar-SO-02'
# dff.loc[dff['TSI NAME'] == 'SATHISH', 'Position'] = 'Kar-SO-07'
# dff.loc[dff['TSI NAME'] == 'NARASIMHARAJU', 'Position'] = 'Kar-SO-09'
# dff.loc[dff['TSI NAME'] == 'RAMESH', 'Position'] = 'Kar-SO-05'
dff = dff.merge(df2, on='Outlet Erp Id', how='left')
dynamic_filters = DynamicFilters(dff, filters=["Territory","Final_Beats","Is Active","Pending_Status"])    
dynamic_filters.display_filters(location='sidebar')
df = dynamic_filters.filter_df()

unique_names = df['Final_Beats'].unique().tolist()
t=0
for name in unique_names:
    filtered_dff2 = df3[df3['Final_Beats'] == name]    
    if not filtered_dff2.empty:  
        t += filtered_dff2['Total_Pending_Amount'].sum()

formatted_amount4 = format_currency(t, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
center_lat = np.mean(df['LAT'])
center_lon = np.mean(df['LONG'])
# filtered_df['Combined Category'] = filtered_df['Brand Presence'].astype(str) + ' | ' + filtered_df['Milk Products SKUs'].astype(str)
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

# Fix marker style (without 'line' property)
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
# st.write(beatwise_pending_amount)
# Display the map in Streamlit
total_rows = dff.shape[0]
total_rows1=df.shape[0]


outlet_counts = df['Beats'].value_counts()
outlet_counts1 = dff['Final_Beats'].value_counts()
pending_outlets_count=df[df["Pending_Status"]=="Pending"].shape[0]
# st.write(outlet_counts)
# st.write(outlet_counts1)
# outlet_counts1 = df['MyBeat Plan'].value_counts()
# grouped_dp = dp.groupby('Particulars_number', as_index=False)['Sum of Diff'].sum()

# # Step 3: Rename the column to 'Total Pending'
# grouped_dp.rename(columns={'Sum of Diff': 'Total Pending'}, inplace=True)

# Display the grouped data
# print(grouped_dp)
non_null_count = int(result['Total Pending'].sum())
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
    st.metric(label="Matched records pending amount", value=str(formatted_amount))
with col2:
    st.metric(label="unmatched records pending amount ", value=str(formatted_amount2))
with col3:
    st.metric(label="Universal outlets", value=f"{total_rows:,}")
# with col4:
#     st.metric(label="Possible duplicates", value=str("(Developing)"))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Beat outlets", value=f"{total_rows1:,}")
with col2:
    st.metric(label="Pending outlets", value=f"{pending_outlets_count:,}")
with col3:
    st.metric(label="TOTAL pending amount / beat", value=str(formatted_amount4))
with col4:
    st.metric(label="Current pending amount / beat", value=str(formatted_amount3))    
    

st.plotly_chart(fig)
# st.write(df)
current_date = datetime.now().date()
formatted_date = current_date.strftime("%d/%m/%y")
b= (
    dff.groupby(["Final_Beats"])["Total Pending"]
    .sum()
    .reset_index()
    .rename(columns={"Total Pending": f"Total Pending({formatted_date})"})
)
# b= b.rename(columns={"final_beat_plan": "Final_Beats"})
# st.write(b)
selected_beats = st.multiselect(
    "Select Beat Names:",
    options=dff["Final_Beats"].unique(),  
    default=[]  
)
# dff = dff.merge(grouped_dp, on='Final_Beats', how='left')
d=df3.merge(b,on='Final_Beats', how='left')
d.index = range(1, len(d) + 1)  # Set index starting from 1
d.index.name = "Beat No"
# Calculate percentage difference
# d['Percent_diff'] = ((d['Total_Pending_Amount'] - d['curent_total_Pending_Amount']) / d['curent_total_Pending_Amount']) * 100
# d['Percent_diff'] = d['Percent_diff'].apply(lambda x: f"{x:.2f}%")
# st.write(selected_beats)
d = d.rename(columns={"Total_Pending_Amount": "Total Pending(11/01/25)"})
if selected_beats: 
    fi = dff[dff["Final_Beats"].isin(selected_beats)]
    pending_outlets_df = fi[fi["Pending_Status"] == "Pending"]

    # Select only the required columns
    pending_outlets_df = pending_outlets_df[["Final_Beats", "Outlets Name", "Total Pending(11/01/25)", "Total Pending"]]
    pending_outlets_df = pending_outlets_df.rename(columns={"Total Pending": f"Total Pending({formatted_date})"})
    # Display the filtered DataFrame in Streamlit
    pending_outlets_df.index = range(1, len(pending_outlets_df) + 1)  # Set index starting from 1
    pending_outlets_df.index.name = "SI No"
    st.write(pending_outlets_df)
else:
    st.write(d)
# col1, col2 = st.columns(2)
# with col1:
#     st.dataframe(outlet_counts)
#     st.write(outlet_counts.count()) 
# with col2:

#     st.dataframe(outlet_counts1) 
#     st.write(outlet_counts1.count())   
# with col2:
# st.write(result) # erpid with amount to be paid 
# filepath = "C:\\Users\\DELL\\Desktop\\gusto\\final_gusto11022025.xlsx"

# df.to_excel(filepath, index=False)
# st.write(df)# overall
# st.write(dp) #coolectins
# dff = dff.drop_duplicates(subset="Outlets Name", keep="first")
# new_df = dp[dp['Particulars_number'].isnull() | (dp['Particulars_number'] == '')]
# new_df = new_df.reset_index(drop=True)
# new_df["Particulars_number"] = new_df["Particulars"].map(
#     dff.set_index("Outlets Name")["Particulars_number"]
# )
# st.write(new_df)
# non_null_count1 = new_df['Sum of Diff'].sum()
# st.metric(label="pending", value=str(non_null_count1))

# import streamlit as stwr
# import pandas as pd
# import folium
# from folium.plugins import MarkerCluster
# import numpy as np
# from streamlit_dynamic_filters import DynamicFilters

# # Set up Streamlit page configuration
# st.set_page_config(
#     page_title='GTA DASHBOARD',          
#     page_icon=None,           
#     layout="wide",        
#     initial_sidebar_state="auto"  
# )

# # Read data function
# @st.cache_data
# def read_data(file):
#     """Read the data from an Excel file."""
#     df = pd.read_excel(file)
#     return df

# # Try loading data from 'gusto.xlsx' file
# try:
#     dff = read_data('gusto.xlsx')
# except FileNotFoundError:
#     st.error("The file 'gusto.xlsx' was not found. Please check the file path and try again.")
#     st.stop()

# # Initialize Dynamic Filters
# dynamic_filters = DynamicFilters(dff, filters=["Territory", "Beats"])

# # Display dynamic filters in the sidebar
# dynamic_filters.display_filters(location='sidebar')

# # Apply the filters to the DataFrame
# df = dynamic_filters.filter_df()

# # Convert Latitude and Longitude to float
# df['LAT'] = df['Latitude'].astype(float)
# df['LONG'] = df['Longitude'].astype(float)
# df.dropna(subset=['LAT', 'LONG'], inplace=True)
# # Calculate the center of the map
# center_lat = np.mean(df['LAT'])
# center_lon = np.mean(df['LONG'])

# Create a folium map centered around the average latitude and longitude
# m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# # Add a MarkerCluster to the map for clustering markers
# marker_cluster = MarkerCluster().add_to(m)

# # Add markers to the map based on the filtered data
# for index, row in df.iterrows():
#     lat = row['LAT']
#     lon = row['LONG']
#     # outlet_name = row['Name of Outlet']
#     # type_of_outlet = row['Type of Outlet']
#     # territory = row['Territory']

#     # Create a popup with the outlet information
#     # popup_info = f"<strong>Outlet:</strong> {outlet_name}<br><strong>Type:</strong> {type_of_outlet}<br><strong>Territory:</strong> {territory}"

#     # Add marker to the cluster
#     folium.Marker(
#         location=[lat, lon],
#         # popup=popup_info,
#         icon=folium.Icon(color='blue')  # You can change the color or use custom icons
#     ).add_to(marker_cluster)

# # Display the map in Streamlit
# st.write("### Outlet Locations Map")
# st.components.v1.html(m._repr_html_(), height=600)

# Display filtered DataFrame and metrics
# total_rows = dff.shape[0]
# total_rows1 = df.shape[0]

# col1, col2 = st.columns(2)
# with col1:
#     st.metric(label="Total Outlets", value=str(total_rows))
# with col2:
#     st.metric(label="Filtered Outlets", value=str(total_rows1))

# # Optionally display filtered DataFrame
# st.write(df)

# filtered_df=df
# b= (
#     dff.groupby(["Final_Beats"])["Total Pending"]
#     .sum()
#     .reset_index()
#     .rename(columns={"Total Pending": "curent_total_Pending_Amount"})
# )
# # b= b.rename(columns={"final_beat_plan": "Final_Beats"})
# st.write(b)
# ndf = beatwise_pending_amount
# filepath = "C:\\Users\\DELL\\Desktop\\gusto\\Gusto_my_record.xlsx"

# df.to_excel(filepath, index=False)
# filtered_df["Lat_Long_Sum"] = filtered_df["Latitude"] + filtered_df["Longitude"]

# # Sort the DataFrame by Lat_Long_Sum
# filtered_df = filtered_df.sort_values("Lat_Long_Sum").reset_index(drop=True)
# # st.write(filtered_df)
# # Check the length of the DataFrame
# if len(filtered_df) > 500:
#     st.warning("The data contains more than 500 records. Mapping is skipped.")
# else:
#     # Initialize a single map
#     map_center = [filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()]
#     m = folium.Map(location=map_center, zoom_start=14)

#     # Group by Beat
#     grouped = filtered_df.groupby("Final_Beats")
    
#     for beat, df in grouped:
#         coordinates = df[["Latitude", "Longitude"]].values

#         # Skip if only one record
#         if len(df) <= 1:
#             st.warning(f"Not enough points to create a route for {beat}.")
#             continue

#         # Calculate a nearest-neighbor route from the sorted data
#         def sorted_route(coords):
#             n = len(coords)
#             visited = [False] * n
#             route = [0]  # Start at the first point
#             visited[0] = True

#             for _ in range(n - 1):
#                 last = route[-1]
#                 distances = cdist([coords[last]], coords)[0]
#                 distances[visited] = np.inf  # Ignore already visited points
#                 next_point = np.argmin(distances)
#                 route.append(next_point)
#                 visited[next_point] = True

#             return route

#         # Optimize the route
#         route = sorted_route(coordinates)
#         optimized_df = df.iloc[route]

#         # Add markers and numbers for each point
#         for i, (idx, row) in enumerate(optimized_df.iterrows()):
#             color = "blue"
#             if i == 0:
#                 color = "red"  # Starting point
#             elif i == len(optimized_df) - 1:
#                 color = "green"  # Ending point
            
#             folium.Marker(
#                 location=[row["Latitude"], row["Longitude"]],
#                 popup=row["Outlets Name"],
#                 tooltip=f"{row['Outlets Name']} ({i+1}, {beat})",
#                 icon=folium.Icon(color=color, icon="info-sign")
#             ).add_to(m)

#         # Add a polyline to connect the route
#         folium.PolyLine(
#             locations=optimized_df[["Latitude", "Longitude"]].values,
#             color="blue" if beat == "Beat1" else "purple",  # Different color for each beat
#             weight=5,
#             opacity=0.7,
#             tooltip=f"Route for {beat}"
#         ).add_to(m)

#     # Display the map in Streamlit
#     st_folium(m, width=900, height=600)