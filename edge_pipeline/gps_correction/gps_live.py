import serial
import pynmea2
import time

port = "/dev/ttyAMA0"
baud = 9600

print("--- Live GPS Tracking Started ---")
try:
    serial_port = serial.Serial(port, baudrate=baud, timeout=1)

    while True:
        line = serial_port.readline().decode('utf-8', errors='ignore')
        if line.startswith('$GPRMC'):
            try:
                msg = pynmea2.parse(line)
                if msg.status == 'A':
                    print(f"Time (UTC): {msg.timestamp} | Lat: {msg.latitude:.6f} | Lon: {msg.longitude:.6f} | Speed: {msg.spd_over_grnd:.1f} knots")
                else:
                    print("Searching for stable satellite connection...")
            except pynmea2.ParseError:
                continue
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nTracking suspended.")
except Exception as e:
    print(f"Hardware Error: {e}")
