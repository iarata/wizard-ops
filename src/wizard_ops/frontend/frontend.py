import logging
import os
from typing import Optional

import requests
import streamlit as st
from google.cloud import run_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_backend_url() -> Optional[str]:
    """Use WIZARD_BACKEND env var if defined, otherwise find backend URL from Google"""
    if os.environ.get("WIZARD_BACKEND"):
        return os.environ.get("WIZARD_BACKEND")
    else:
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
    """Display 2 columns, left for image upload, right for results. Indicate loading/analysis with a spinner"""
    backend = get_backend_url()
    if backend is None:
        raise ValueError("Backend service not found")

    st.set_page_config(
        page_title="Food Image Analyzer", layout="wide", page_icon=":hamburger:"
    )
    st.title("🍔 Food Image Analyzer")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Upload food image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    with col2:
        st.header("Results")

        if uploaded_file:
            with st.spinner("Analyzing..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                try:
                    response = requests.post(f"{backend}/analyze", files=files)
                    logger.info(f"response {response.json()}")

                    response.raise_for_status()
                    st.session_state["result"] = response.json()
                except requests.exceptions.ConnectTimeout:
                    logger.exception("Backend connection timed out")
                    st.error(
                        body="Backend connection timed out.",
                        icon="⌛"
                    )
                except requests.exceptions.ConnectionError:
                    logger.exception("Backend is unreachable")
                    st.error(
                        body="Cannot connect to the backend service",
                        icon="⚠️"
                    )
                except requests.exceptions.HTTPError as e:
                    logger.exception(f"Backend returned an HTTP error {e}")
                    try:
                        error_detail = response.json()
                    except Exception:
                        error_detail = response.text

                    st.error(
                        body=f"Server returned an error ({response.status_code})"
                             f"{error_detail}",
                        icon="🔴"
                    )
                except requests.exceptions.RequestException:
                    logger.exception("Unexpected request error")
                    st.error(
                        "An unexpected error occurred while contacting the server.",
                        icon="🔴"
                    )

        if "result" in st.session_state:
            result = st.session_state["result"]

            st.metric("Calories, kcal", f'{result["calories"]}', format="%0.2f")
            st.metric("Fat, g", f'{result["fat_g"]}', format="%0.2f")
            st.metric("Protein, g", f'{result["protein_g"]}', format="%0.2f")
            st.metric("Carbs, g", f'{result["carbs_g"]}', format="%0.2f")
        else:
            st.info("Analysis results will show up here")


if __name__ == "__main__":
    main()
