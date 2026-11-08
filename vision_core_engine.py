import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

st.set_page_config(
    page_title="AI Vision & Emotion Tracker",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Computer Vision & Smart Face Emotion Tracker")
st.markdown("---")

st.sidebar.header("⚙️ Control Panel")
app_mode = st.sidebar.selectbox(
    "Choose Mode / इनपुट मोड चुनें",
    ["Live Webcam Stream", "Upload Image/Video"]
)

def analyze_frame(frame):
    try:
        analysis = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=False
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
            
        emotion = analysis.get('dominant_emotion', 'Neutral')
        region = analysis.get('region', {})
        
        return emotion, region
    except Exception as e:
        st.sidebar.warning(f"Smart Warning: Face processing adjustment active ({str(e)})")
        return "Neutral", {}

if app_mode == "Live Webcam Stream":
    st.subheader("🔴 Live Webcam Emotion Tracking")
    st.write("मैबुक या डिवाइस के कैमरे से रियल-टाइम इमोशन ट्रैकिंग चालू करने के लिए नीचे दिए गए बटन पर क्लिक करें।")
    
    run_webcam = st.checkbox("Start Live Camera / कैमरा शुरू करें")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_emotion = st.metric(label="Detected Emotion / भाव", value="Waiting...")
    with col2:
        metric_status = st.metric(label="Tracker Status / स्थिति", value="Standby")
    with col3:
        metric_fps = st.metric(label="System Mode", value="Zero-Crash Active")

    frame_placeholder = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        metric_status.metric(label="Tracker Status / स्थिति", value="Active / चल रहा है")
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("स्मार्ट वार्निंग: कैमरा फ़ीड प्राप्त करने में असमर्थ। कृपया कैमरा कनेक्शन जांचें।")
                break
                
            dominant_emotion, region = analyze_frame(frame)
            
            if region and all(k in region for k in ('x', 'y', 'w', 'h')):
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"Emotion: {dominant_emotion}", 
                    (x, max(y - 10, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 255, 0), 
                    2
                )
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            metric_emotion.metric(label="Detected Emotion / भाव", value=dominant_emotion.capitalize())
            
        cap.release()
    else:
        st.info("कैमरा बंद है। शुरू करने के लिए ऊपर दिए गए चेकबॉक्स पर टिक करें।")

elif app_mode == "Upload Image/Video":
    st.subheader("📁 Image / File Upload Vision Matrix")
    uploaded_file = st.file_uploader(
        "अपनी इमेज या मीडिया फाइल यहाँ अपलोड करें (Supports: JPG, PNG, JPEG)", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            if opencv_image is None:
                raise ValueError("अमान्य फाइल फॉर्मेट या करप्ट इमेज डाटा।")
                
            dominant_emotion, region = analyze_frame(opencv_image)
            
            if region and all(k in region for k in ('x', 'y', 'w', 'h')):
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    opencv_image, 
                    f"Emotion: {dominant_emotion}", 
                    (x, max(y - 10, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (255, 0, 0), 
                    2
                )
                
            st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            st.success(f"सफलतापूर्वक डिटेक्ट किया गया! मुख्य भाव (Dominant Emotion): **{dominant_emotion.capitalize()}**")
            
        except Exception as e:
            st.warning(f"⚠️ स्मार्ट वार्निंग: फाइल प्रोसेस करते समय समस्या आई ({str(e)})। कृपया दूसरी फाइल आज़माएं।")
    else:
        st.info("कृपया आगे बढ़ने के लिए कोई इमेज अपलोड करें।")

st.markdown("---")
st.markdown("✨ *AI Computer Vision & Smart Face Emotion Tracker - Built with Python, OpenCV & Streamlit.*")
