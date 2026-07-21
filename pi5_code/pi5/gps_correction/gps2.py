import serial, pynmea2, json
from datetime import datetime, timezone

GPS_PORT = "/dev/ttyAMA0"
GPS_BAUD = 9600
MIN_SATS = 4
MAX_HDOP = 3.0

class Kalman1D:
    def __init__(self, Q=1e-5, R=0.005):
        self.Q = Q
        self.R = R
        self.P = 1.0
        self.x = None

    def update(self, z, hdop=1.0):
        if self.x is None:
            self.x = z
            return self.x
        self.P += self.Q
        R_scaled = self.R * (hdop ** 2)
        K = self.P / (self.P + R_scaled)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x

def run(output_file="gps_log.jsonl"):
    ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
    lat_k = Kalman1D()
    lon_k = Kalman1D()
    alt_k = Kalman1D(Q=1e-4, R=0.1)
    fix_count = 0
    skip_count = 0

    print(f"Logging to {output_file} | Ctrl+C to stop\n")

    with open(output_file, "a") as f:
        for raw in ser:
            try:
                line = raw.decode("ascii", errors="ignore").strip()
                msg = pynmea2.parse(line)

                if not isinstance(msg, pynmea2.GGA):
                    continue

                fix  = int(msg.gps_qual or 0)
                sats = int(msg.num_sats or 0)
                hdop = float(msg.horizontal_dil or 99)
                lat  = msg.latitude
                lon  = msg.longitude
                alt  = float(msg.altitude or 0)

                if fix == 0 or lat == 0.0:
                    skip_count += 1
                    print(f"  [no fix] sats={sats}", end="\r")
                    continue

                if sats < MIN_SATS:
                    skip_count += 1
                    print(f"  [skip] only {sats} sats", end="\r")
                    continue

                # ✅ hdop now passed correctly
                lat_f = lat_k.update(lat, hdop)
                lon_f = lon_k.update(lon, hdop)
                alt_f = alt_k.update(alt, hdop)
                fix_count += 1

                record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "lat": round(lat_f, 7),
                    "lon": round(lon_f, 7),
                    "alt": round(alt_f, 2),
                    "lat_raw": round(lat, 7),
                    "lon_raw": round(lon, 7),
                    "hdop": hdop,
                    "sats": sats,
                    "fix_type": fix,
                    "n": fix_count
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                print(f"[{fix_count:04d}] {lat_f:.7f}, {lon_f:.7f}  "
                      f"alt={alt_f:.1f}m  hdop={hdop:.1f}  sats={sats}  "
                      f"fix={'DGPS' if fix==2 else 'GPS'}")

            except (pynmea2.ParseError, ValueError):
                pass
            except KeyboardInterrupt:
                print(f"\n\nDone. {fix_count} fixes logged, {skip_count} skipped.")
                break

if __name__ == "__main__":
    run()
