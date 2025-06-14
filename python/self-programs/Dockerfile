FROM python:3.7-slim

# Install the API wheels and other requirements.
COPY requirements.txt prebuilt/*.whl ./

RUN python3 -m pip install -r requirements.txt --find-links .

COPY basic_program.py /
#COPY network_request_service_manager.py /

WORKDIR /

ENTRYPOINT ["python3", "basic_program.py"]
# Default arguments for running on the Spot CORE
CMD [ "192.168.50.3", "--payload-credentials-file=/creds/payload_guid_and_secret"]