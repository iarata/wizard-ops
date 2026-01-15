# gcloud auth configure-docker

docker build --platform linux/amd64 -t gcr.io/dtumlops-484413/streamlit-app -f dockerfiles/frontend.dockerfile .

docker push gcr.io/dtumlops-484413/streamlit-app

gcloud run deploy streamlit-app \
    --image gcr.io/dtumlops-484413/streamlit-app \
    --region europe-west1 \
    --allow-unauthenticated

# Service [streamlit-app] revision [streamlit-app-00001-rjw] has been deployed and is serving 100 percent of traffic.
# Service URL: https://streamlit-app-1043637954808.europe-west1.run.app
#
# ... of course, it was neither AMD64/ARM64, nor PORT, nor anything else...
# ... unless all this time docker did not change the contents... but only the metadata... doubt... lets try a new image with dockerhub :)
# ... you'd also think ARM64 would be supported at this point