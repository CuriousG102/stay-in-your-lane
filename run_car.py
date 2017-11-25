import sim_client
import time
S = sim_client.SimClient()
S.start()
time.wait(5)
while True:
    S.get_telemetry()
    S.send_instructions(1,1)
S.stop()
