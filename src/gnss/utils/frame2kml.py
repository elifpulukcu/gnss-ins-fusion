# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:21:14 2025

@author: sorla
"""

import os
import numpy as np
import pandas as pd

pi_gps = 3.1415926535898 #Pi in the gps system
a = 6378137.000 #Semi-major axis
f = 1/298.257223563 #falttening
b = -f*a+a #6356752.314 - semi-minor axis;
esq = (a**2-b**2)/a**2 #squared eccentricity, e^2
ohm_e = 0.7292115e-4 #Earth's rotation rate

def make_placemark(row,tStamp, altitudeMode="clampToGround", useGoogleHeights=False):
    # name = str(row.get('name', 'Placemark'))
    lat = row['latitude']
    lon = row['longitude']
    alt = row.get('altitude', 0)
    
    if useGoogleHeights:
        alt = max(0, alt-40)
    
    coord_str = f"{lon},{lat},{alt}"
    timestamp = tStamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    tname = tStamp.strftime("%H:%M:%S")
    
    return f"""
    <Placemark>
        <TimeStamp>
          <when>{timestamp}</when>
        </TimeStamp>
        <name>{tname}</name>
        <styleUrl>#mystyle</styleUrl>
        <Point>
            <coordinates>{coord_str}</coordinates>
            <altitudeMode>{altitudeMode}</altitudeMode>
        </Point>
    </Placemark>
    """
    
def make_linestring(row, tStamp, useGoogleHeights=False):
    
    # name = str(row.get('name', 'Placemark'))
    lat = row['latitude']
    lon = row['longitude']
    alt = row.get('altitude', 0)
    
    if useGoogleHeights:
        alt = max(0, alt-40)
    
    coord_str = f"{lon},{lat},{alt}"
    # timestamp = tStamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    # tname = tStamp.strftime("%H:%M:%S")
    
    return f"""{coord_str}"""

def dataframe_to_kml(dfin, filename="output.kml", altitudeMode="clampToGround", color="green", placeorline=3, useGoogleHeights=False):
    """
    Converts a pandas DataFrame with geodetic positions into a KML file for Google Earth.
    
    Required columns: 'latitude', 'longitude'
    Optional columns: 'name', 'altitude'
    
    color: 
        Purple: 0xFF55007F
        Green: 0xFF00FF00
        Yellow: 0xFFFFFF00
        Red: 0xFFFF0000
        Blue: 0xFF0000FF
        Orange: 0xFFFFAA00
        Teal: 0xFF55FFFF
        Pink: 0xFFFF55FF
        White: 0xFFFFFFFF
    
    Parameters:
        df (pd.DataFrame): DataFrame with geodetic data.
        filename (str): Output filename for the KML file.
        altitudeMode (str): one of:
            clampToGround
            absolute
            relativeToGround

        useGoogleHeights: Boolean
            Note! googleHeights tries to accomodate for apparent offset
            
    """
    
    df = cart2geo(dfin.iloc[:3,:].values)
    df = pd.DataFrame(df, index=["longitude", "latitude", "altidtude"], columns=dfin.columns)
    df = df.T
    
    colorDict = {
        "purple": "0xFF7F0055",
        "green": "0xFF00FF00",
        "yellow": "0xFF00FFFF",
        "red": "0xFF0000FF",
        "blue": "0xFFFF0000",
        "orange": "0xFF00AAFF",
        "teal": "0xFFFFFF55",
        "pink": "0xFFFF55FF",
        "white": "0xFFFFFFFF",
        }
    
    # Make sure color is lower case
    color = color.lower()
    
    # select color
    if color in colorDict:
        color = colorDict[color]
    else:
        # Default to green
        color = 0xFF00FF00
    
    # Set altitude mode
    if altitudeMode not in ["clampToGround", "absolute", "relativeToGround"]:
        altitudeMode = "clampToGround"

    # Generate header
    kml_header = f"""<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
        <Document>
            <name>GPS track</name>
                <Style id="mystyle">
            	  <IconStyle>
            		<scale>1.0</scale>
                    <color>{color}</color>
            		<Icon>
            		  <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
            		</Icon>
            	  </IconStyle>
            	</Style>
        """
    
    folder1 = """<Folder>
        <name>Placemarks</name>
    """
    
    folder2 = """<Folder>
		<name>GPS trace</name>
    """
    
    folderend = """</Folder>"""
    
    linestyleid = f"""<Placemark>
        <name>"GPS position line"</name>
            <Style>
				<LineStyle>
					<color>{color}</color>
					<width>4.0</width>
				</LineStyle>
			</Style>
			<MultiGeometry>
				<LineString>
					<altitudeMode>{altitudeMode}</altitudeMode>
                        <coordinates>
    """
    
    linestyleend = """</coordinates>
					<tessellate>1</tessellate>
				</LineString>
			</MultiGeometry>
		</Placemark>
    """
    
    kml_footer = """</Document>
        </kml>"""
    
    # Check if we need placemarker, linestring or both    
    if placeorline==3:
        placemarks = [make_placemark(row, tStamp, altitudeMode, useGoogleHeights) for tStamp, row in df.iterrows()]
        line = [make_linestring(row, tStamp, useGoogleHeights) for tStamp, row in df.iterrows()]
        kml_content = kml_header + folder1 + "".join(placemarks) + folderend + folder2 + linestyleid + "\n".join(line) + linestyleend + folderend  + kml_footer
    elif placeorline==2:
        line = [make_linestring(row, tStamp, useGoogleHeights) for tStamp, row in df.iterrows()]
        kml_content = kml_header + folder2 + linestyleid + "\n".join(line) + linestyleend + folderend  + kml_footer
    else:
        placemarks = [make_placemark(row, tStamp, altitudeMode, useGoogleHeights) for tStamp, row in df.iterrows()]
        kml_content = kml_header + folder1 + "".join(placemarks) + folderend + kml_footer
    
    # Ensure correct filename
    if '.' in filename:
        if filename.split('.')[-1] != 'kml':
            filename = filename+".kml"
    else:
        filename = filename+".kml"
    
    directory = filename.rpartition("/")[0]
    
    if any(directory) and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(kml_content)
    
    print(f"KML file successfully written to: {filename}")

def cart2geo(coord, ipph=0):
    #Converts positions from Carteesian coordinates to Geodetic Coordinates (in degrees!)
    #Input: 
    #coord array of size [3xNumberOfPositions], where: X is first row, Y is second row, Z is third row
    #ipph which is used for ipp calculations, the height of the ionosphere in meters
    #Output:
    #P - [long, lat, heigh] in [deg] and [m]
    
    #Store coordinates
    if np.ndim(coord)>1:
        X=coord[0,:]; Y=coord[1,:]; Z=coord[2,:]; #if input is a matrix
        szc=np.size(coord,axis=1)
    else:
        X=coord[0]; Y=coord[1]; Z=coord[2]; #if only one dimesion
        szc=1

    h = np.zeros([szc])+40 #initial guess of height
    h_old = np.zeros([szc]) #initial 'old' height (used for first calc of dh)
    dh = np.zeros([szc])+5 #initial dh value (arbitrary, but must be larger than 0.001)
    
    N = 6385000 # Initial radius of prime vertical 
    
    #Define new Earth with ipp height
    a = 6378137.000+ipph #Semi-major axis
    # f = 1/298.257223563 #flattening
    f = 1/298.257222101 #flattening (ETRS89)
    b = -f*a+a #6356752.314 - semi-minor axis;
    esq = (a**2-b**2)/a**2 #squared eccentricity, e^2
    
    #Calculate longitude
    long = np.arctan2(Y, X)
    
    #Currently NO check for North or South Pole!!
    
    #Calculate latitude and height
    while (dh > 0.0005).all():
        lat = np.arctan(Z/np.sqrt(X**2 + Y**2)*(1-esq*N/(N+h))**(-1))
        h = np.sqrt(X**2+Y**2)/np.cos(lat) - N

        #Check "precision" of iteration
        N = a/np.sqrt(1-esq*np.sin(lat)**2) #Radius of prime vertical
        dh = np.abs(h_old - h) #height difference
        h_old = h #store h for next iteration
    
    if szc == 1:
        lat = lat[0]
        h = h[0]
    
    # print(long)
    # print(lat)
    # print(h)
    
    P = np.array([long/pi_gps*180, lat/pi_gps*180, h], dtype=float)
    
    return P