# 🚗 Real-Time Drowsiness Detection System

A real-time driver drowsiness detection system using **YOLOv5, PyTorch, and OpenCV** with a **Streamlit web interface**.

## 🔥 Features
- Real-time video stream monitoring
- Drowsiness detection using custom-trained YOLOv5 model
- Alarm system when drowsiness detected
- Streamlit UI (Start/Stop camera)

## 🛠️ Tech Stack
- Python
- PyTorch
- YOLOv5
- OpenCV
- Streamlit

## 📁 Project Structure
- app.py
- RTDD-checkpoint
- best.pt
- alarm.mpeg
- requirements.txt

## 📊 Dataset
- Custom dataset labeled using LabelImg

## 💡 Future Improvements
- Improve accuracy
- Deploy on cloud
- Mobile integration
 
 
## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py