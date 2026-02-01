# write code to loop through every circuit in the data/hackathon_public.json file
# and make a matlotlib line graph of forward runtime vs threshold rung for each circuit's threshold sweep
# save the graph to a file called rung_trends.png

import json
import matplotlib.pyplot as plt

with open("data/hackathon_public.json", "r") as f:
    data = json.load(f)

plt.scatter([x['forward']['threshold'] for x in data["results"]], [x['forward']['run_wall_s'] for x in data["results"]])
plt.xlabel("Threshold")
plt.ylabel("Runtime (s)")
plt.show()