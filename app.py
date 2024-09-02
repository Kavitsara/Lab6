import torch

# สมมติว่า ModelClass เป็นคลาสของโมเดลที่คุณใช้
class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        # กำหนดโครงสร้างของโมเดลที่นี่

# โหลดโมเดล
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = torch.load('model.pth', map_location=device)

    if isinstance(model, dict):  # ถ้า model เป็น state_dict
        model_instance = ModelClass()  # สร้าง instance ของโมเดล
        model_instance.load_state_dict(model)
        model = model_instance

    model = model.to(device)  # ย้ายโมเดลไปยัง device
    model.eval()  # ตั้งโมเดลให้เป็นโหมดประเมินผล

except Exception as e:
    print(f"Error loading model: {str(e)}")
