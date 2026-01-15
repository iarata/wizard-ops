import streamlit as st
import requests

from typing import Optional

from google.cloud import run_v2
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_backend_url() -> Optional[str]:
    parent = "projects/dtumlops-484413/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)

    logger.info(f"services {services}")

    for service in services:
        logger.info(f"service {service}")
        if service.name.split("/")[-1] == "wizard-ops-api":
            return service.uri

    return None


def main():
    # Prioritize override from env var
    backend = os.environ.get("WIZARD_BACKEND") # or get_backend_url()
    if backend is None:
        raise ValueError("Backend service not found")

    st.set_page_config(page_title="Food Calorie Analyzer", layout="wide", page_icon=":hamburger:")
    st.title("🍔 Food Image Analyzer")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Upload food image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            # if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(f"{backend}/analyze", files=files)
                logger.info(f"response {response.json()}")

                if response.status_code == 200:
                    st.session_state["result"] = response.json()
                else:
                    st.error("Error from FastAPI")

    with col2:
        st.header("Results")

        if "result" in st.session_state:
            result = st.session_state["result"]

            st.metric("Calories", f'{result["calories"]} kcal')
            st.metric("Fat", f'{result["fat_g"]} g')
            st.metric("Protein", f'{result["protein_g"]} g')
            st.metric("Carbs", f'{result["carbs_g"]} g')
        else:
            st.info("Analysis results will show up here")


if __name__ == "__main__":
    main()
