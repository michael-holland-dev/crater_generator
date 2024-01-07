from Dataset import CraterDataset

dataset = CraterDataset(500, "./data/")

planet, body = dataset[2]
 
print(planet)
