import socket
import subprocess
import os
import signal

UDP_IP = "0.0.0.0"  # Écoute sur le Wi-Fi ET l'Ethernet Skydroid
UDP_PORT = 5005

# Remplacer par le nom exact du fichier de votre application principale
SCRIPT_PRINCIPAL = "/home/pi5/Documents/real_test/stress6.py" 
processus_application = None

def recuperer_pid(nom_script):
    """Retrouve le PID du script s'il tourne déjà."""
    try:
        pid = subprocess.check_output(["pgrep", "-f", nom_script]).decode().strip().split('\n')[0]
        return int(pid)
    except Exception:
        return None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"👂 Écoute des commandes Skydroid sur le port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(1024)
    commande = data.decode('utf-8').strip().upper()
    print(f"Commande reçue de {addr} : {commande}")
    
    pid = recuperer_pid(SCRIPT_PRINCIPAL)
    
    if commande == "START":
        if pid:
            print("▶️ L'application tourne déjà. Envoi du signal de reprise (CONT)...")
            os.kill(pid, signal.SIGCONT)
        else:
            print("🚀 L'application n'est pas lancée. Démarrage initial...")
            # Lance l'application en tâche de fond
            # Ligne correcte à intégrer dans listen_skydroid.py
            subprocess.Popen(["/home/pi5/Documents/venv/bin/python3", "-u", SCRIPT_PRINCIPAL], cwd="/home/pi5/Documents")


            
    elif commande == "PAUSE":
        if pid:
            print("⏸️ Application trouvée (PID: {}). Mise en pause (STOP)...".format(pid))
            os.kill(pid, signal.SIGSTOP)
        else:
            print("❌ Impossible de mettre en pause : l'application ne tourne pas.")
