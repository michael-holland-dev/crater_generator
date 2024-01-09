from Dataset import CraterDataset
from PIL import Image

dataset = CraterDataset(500, "./data/")

planet, crater = dataset[2]
 
print(planet)

crater.show()