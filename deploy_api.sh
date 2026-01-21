docker build --platform linux/amd64 -t gcr.io/dtumlops-484413/wizard-ops-api -f dockerfiles/api.dockerfile .

docker push gcr.io/dtumlops-484413/wizard-ops-api

gcloud run deploy wizard-ops-api \
    --image gcr.io/dtumlops-484413/wizard-ops-api \
    --region europe-west1 \
    --memory 2Gi \
    --scaling=1 \
    --allow-unauthenticated
#    --cpu 2

# startup: Memory limit of 512 MiB exceeded with 516 MiB used. Consider increasing the memory limit, see https://cloud.google.com/run/docs/configuring/memory-limits
# running, loading model: Memory limit of 1024 MiB exceeded with 1058 MiB used. Consider increasing the memory limit, see https://cloud.google.com/run/docs/configuring/memory-limits
# => setting to 2gigs worked {"status":"ready","model_loaded":true}

# Service [wizard-ops-api] revision [wizard-ops-api-00010-ljg] has been deployed and is serving 100 percent of traffic.
# Service URL: https://wizard-ops-api-1043637954808.europe-west1.run.app