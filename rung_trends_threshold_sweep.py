# write code to loop through every circuit in the data/hackathon_public.json file
# and make a matlotlib line graph of forward runtime vs threshold rung for each circuit's threshold sweep
# save the graph to a file called rung_trends.png

import json
import matplotlib.pyplot as plt

with open("data/hackathon_public.json", "r") as f:
    data = json.load(f)

for circuit in data["results"]:
    threshold_sweep = circuit["threshold_sweep"]
    plt.plot([x['threshold'] for x in threshold_sweep], [x['run_wall_s'] for x in threshold_sweep], label=circuit["file"])
    plt.xlabel("Threshold")
    plt.ylabel("Runtime (s)") # logarithmic
    plt.yscale("log")

plt.legend()
plt.show()