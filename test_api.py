import requests

# URL ของ API ที่คุณต้องการทดสอบ
url = 'http://127.0.0.1:5000/predict'

# เส้นทางของไฟล์รูปภาพที่คุณต้องการใช้ในการทดสอบ
file_path = 'image/im001_n.jpg'  # แก้ไขเส้นทางนี้เป็นที่อยู่ของไฟล์รูปภาพบนเครื่องคุณ

# เปิดไฟล์รูปภาพและส่งไปยัง API
with open(file_path, 'rb') as img_file:
    response = requests.post(url, files={'file': img_file})

# แสดงผลลัพธ์ที่ได้รับจาก API
print(response.json())
