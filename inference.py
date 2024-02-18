import torch
import CTorchClasses as C
from PIL import Image
import torchvision.transforms.v2 as v2
from CapturePalm import CapturePalm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/model0"

def inference(model, input1:torch.Tensor, input2:torch.Tensor, dist_fn_method:str="cosine_l2_norm"):
    output1 = model(input1)
    output2 = model(input2)
    
    obj = C.TripletMarginWithComplexDistanceLoss(dist=dist_fn_method)
    
    output = obj.distance_function(output1, output2)
    
    if dist_fn_method == "cosine_ang":
        output = output.abs()
    
    print("distance = ", output.item())
    pred = "match" if output <= 0.45 else "no_match"
    
    return pred

model = torch.load(MODEL_PATH, map_location=torch.device('cpu')).to(DEVICE)

# MANUALLY READ IMAGES
# input_path_1 = "YOUR/PATH/HERE"
# input_path_2 = "YOUR/PATH/HERE"
# input1 = Image.open(input_path_1)
# input2 = Image.open(input_path_2)

# READ IMAGES FROM WEBCAM
cap = CapturePalm()
input1, input2 = cap.capture()
input1 = torch.from_numpy(input1).unsqueeze(0)
input2 = torch.from_numpy(input2).unsqueeze(0)

transform = v2.Compose([v2.Grayscale() ,v2.Resize((128,128), antialias=True),  v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # last one is transforms.ToTensor()

input1 = transform(input1).unsqueeze(0).to(DEVICE) # B, C, H, W
input2 = transform(input2).unsqueeze(0).to(DEVICE)

print(inference(model, input1, input2))
