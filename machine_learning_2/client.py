
import requests
import pyarrow.feather as feather

X_test = feather.read_feather("./results/X_test.ftr")

sample ={
    "columns":["Customer Type", "Age", "Type of Travel", "Class", "Flight Distance", "Inflight wifi service", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness", "Arrival Delay in Minutes"],
    "data": None
}

df = X_test[:6]

for index, row in df.iterrows():
    sample["data"] = [list(row.to_numpy())]
    response = requests.post("http://127.0.0.1:1234/invocations", json=sample)
    print("Satisfied" if response.json()[0] == 1 else "Insatisfied")