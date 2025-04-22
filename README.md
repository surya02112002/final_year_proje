# Skintellect 🧬✨

![Project Banner](static/assets/bg5.webp)

A intelligent skincare analysis and recommendation system combining computer vision with skincare science.

## Features ✨
- 🧑⚕️ AI-powered skin condition analysis using YOLOv8 object detection
- 💄 Personalized product recommendations
- 📸 Image-based skin assessment
- 📅 Appointment booking system
- 👤 User authentication & profile management

## Tech Stack 🛠️
- **Backend**: Python Flask (app.py)
- **ML Framework**: TensorFlow/Keras (final_model.h5)
- **Object Detection**: Ultralytics YOLOv8 (yolov8n.pt)
- **Database**: SQLite (app.db)
- **Frontend**: HTML5/CSS3 + Jinja2 templating

## Installation ⚙️

```bash
# Clone repository
git clone https://github.com/yourusername/Skintellect.git
cd Skintellect

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements_updated.txt
```

## Usage 🚀

1. Start Flask development server:
```bash
python app.py
```

2. Access web interface at `http://localhost:5000`

3. Key paths:
- `/face_analysis` - Skin image analysis
- `/survey` - Skin questionnaire
- `/recommendations` - Product suggestions

## Dataset 🔢
- `dataset/cosmetics.csv`: 10,000+ skincare products with ingredients
- `dataset/updated_skincare_products.csv`: Curated product recommendations

## Model Architecture 🧠
- Custom CNN for skin analysis (model/final_model.h5)
- YOLOv8n for lesion detection (runs/train32/ weights)

## Deployment 🚀

AWS EC2 Deployment Guide:
1. Launch EC2 instance with Ubuntu 22.04 LTS
2. Configure security groups to allow HTTP/HTTPS traffic
3. Connect via SSH and install requirements:
```bash
sudo apt update && sudo apt install python3-pip nginx
pip install -r requirements_updated.txt
```
4. Configure Nginx reverse proxy for Flask app
5. Set up production WSGI server:
```bash
gunicorn -w 4 app:app
```
6. Configure domain `himanshudixit.info` in Nginx
7. Enable automatic restart with systemd service

Access live at: [https://himanshudixit.info](https://himanshudixit.info)
