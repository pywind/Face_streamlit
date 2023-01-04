import streamlit as st
import tempfile
import predictor
import gdown

from keras.models import load_model

url = 'https://drive.google.com/file/d/1lPWJTBOHQhhi_Up7TtndmJOq9-xMliwr/view?usp=sharing'
output_path = 'model/model.h5'
gdown.download(url, output_path, quiet=False,fuzzy=True)

#load model from file
MODEL = load_model('model/model.h5')

ACCEPTED_PREFIX = ['.mp4', '.avi']

# UI for the App
def run():
    st.title("Violence Detection using OpenCV")
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    with st.sidebar:
        st.markdown("# My Teams")
        st.caption('This is a web app from Phuoc and Nam. A member in Violence Detection Group !!!')

        link = '[©Developed by 19133045-19133037](http://github.com/pywind)'
        st.sidebar.markdown(link, unsafe_allow_html=True)
    st.markdown(
        '''<h4 style='text-align: left; color: #d73b5c;'>* Violence detection with VGG19 + LSTM</h4>''',
        unsafe_allow_html=True)
    vid_file = st.file_uploader("Choose an Video", type=ACCEPTED_PREFIX)
    
    if vid_file is not None:
        with st.spinner('Wait for it...'):
          tfile = tempfile.NamedTemporaryFile(delete=False)
          tfile.write(vid_file.read())
          #vs = cv2.VideoCapture(tfile.name)
          pred = predictor.main_fight(tfile.name, model=MODEL)

          #vs.release()
          tfile.close()
        st.success(f"Result of recognize: {pred['violence']}")
        st.info(f"Violence estimation: {pred['violence estimation']}")
        st.error(f"Time prediction: {pred['processing_time']}")
            # Show video
      
        st.video(vid_file)


if __name__ == "__main__":
    run()
