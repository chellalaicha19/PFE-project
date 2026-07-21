#!/usr/bin/env python3
import serial
import pynmea2
import time
import sys

GPS_PORT = "/dev/ttyAMA0"
GPS_BAUD = 9600

def read_gps(duration=10):
    """Read GPS data for a specified duration."""
    try:
        # Open with exclusive access
        ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1, exclusive=True)
        print(f"Connected to {GPS_PORT} (exclusive mode)")
        
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < duration:
            line = ser.readline()
            if line:
                try:
                    line_str = line.decode("ascii", errors="ignore").strip()
                    if line_str.startswith("$"):
                        # Look for GGA sentences
                        if "GGA" in line_str:
                            msg = pynmea2.parse(line_str)
                            if isinstance(msg, pynmea2.GGA):
                                fix = int(msg.gps_qual or 0)
                                sats = int(msg.num_sats or 0)
                                hdop = float(msg.horizontal_dil or 99)
                                lat = msg.latitude
                                lon = msg.longitude
                                
                                if fix > 0 and lat != 0:
                                    count += 1
                                    print(f"[{count}] LAT:{lat:.6f} LON:{lon:.6f} SATS:{sats} HDOP:{hdop:.1f}")
                except:
                    pass
        
        ser.close()
        print(f"\nTotal fixes: {count}")
        
    except serial.SerialException as e:
        print(f"Error: {e}")
        print("Port might be in use. Try: sudo fuser -k /dev/ttyAMA0")
        sys.exit(1)

if __name__ == "__main__":
    read_gps(20)
